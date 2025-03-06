from __future__ import annotations
from typing import Callable

import math
from functools import partial
from collections import namedtuple

import torch
from torch import nn, cat, Tensor
import torch.nn.functional as F
from torch.nn import Linear, ParameterList
from torch.func import functional_call, vmap, grad

from tensordict import TensorDict

from titans_pytorch.associative_scan import AssocScan
from titans_pytorch.memory_models import MemoryMLP, ResidualNorm

import einx
from einops import rearrange, reduce, repeat

# (1) NamedTuple for neural memory state
NeuralMemState = namedtuple('NeuralMemState', [
    'seq_index',               # 현재까지 처리한 토큰/시퀀스 인덱스
    'weights',                 # 메모리 모델의 파라미터 (M_t)
    'cache_store_segment',     # 다음 스텝에 이어 붙일 시퀀스 캐시
    'states',                  # (past_update, past_momentum) 등
    'updates',                 # 모든 time-chunk별 업데이트(디버깅용)
])

def mem_state_detach(state: NeuralMemState):
    def _detach_if_tensor(x):
        return x.detach() if isinstance(x, Tensor) else x

    return NeuralMemState(*[
        _detach_if_tensor(s) if not isinstance(s, (tuple, dict, TensorDict))
        else (
            s.apply(lambda t: t.detach()) if isinstance(s, TensorDict)
            else s
        )
        for s in state
    ])

# ==========================================
# 도우미 함수들 (기존 코드와 동일하거나 간소화)
# ==========================================

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def round_down_multiple(seq, mult):
    return seq // mult * mult

def round_up_multiple(seq, mult):
    return math.ceil(seq / mult) * mult

def safe_cat(tensors, dim=-2):
    tensors = list(filter(exists, tensors))
    if len(tensors) == 0:
        return None
    return cat(tensors, dim=dim) if len(tensors) > 1 else tensors[0]

# 예시로 간단한 loss(식 (12) 참고)
def default_loss_fn(pred, target):
    # || M(k_t) - v_t ||^2
    return (pred - target).pow(2).mean(dim=-1)

def default_adaptive_step_transform(adaptive_step, max_lr=1e-2):
    # sigmoid로 [0,1] 구간 -> 학습률 upper bound
    return adaptive_step.sigmoid() * max_lr

# ==========================================
# 메인: NeuralMemory 클래스
# ==========================================

class NeuralMemory(nn.Module):
    def __init__(
        self,
        dim,
        chunk_size: int = 1,
        # ------------------
        # 이하 하이퍼파라미터들
        batch_size = None,          # 병렬 업데이트 단위
        dim_head = None,
        heads = 1,
        model: nn.Module | None = None,
        store_memory_loss_fn: Callable = default_loss_fn,
        adaptive_step_transform: Callable | None = None,
        default_step_transform_max_lr = 1.,
        # 모멘텀 / 망각 관련
        momentum = True,
        learned_momentum_combine = False,
        momentum_order = 1,
        # ...
        activation: nn.Module | None = None,
        init_adaptive_step_bias = None,
        init_momentum_bias = None,
        init_decay_bias = None,
        # 논문 MLP 기본
        default_model_kwargs: dict = dict(
            depth = 2,
            expansion_factor = 4.
        ),
    ):
        super().__init__()
        dim_head = default(dim_head, dim)

        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.heads = heads

        # MLP 형태의 메모리 모델
        if not exists(model):
            model = MemoryMLP(dim_head, **default_model_kwargs)
        model = ResidualNorm(dim=dim_head, model=model)
        self.memory_model = model

        # 모델 파라미터 가져오기
        mem_params = dict(model.named_parameters())
        self.memory_model_parameter_names = list(mem_params.keys())
        memory_model_parameters = list(mem_params.values())

        # (가장 간단한 형태) 헤드는 weight를 share 하지 않고 반복
        memory_model_parameters = [
            repeat(p, '... -> h ...', h=heads)
            for p in memory_model_parameters
        ]
        self.init_weight_shape = [p.shape for p in memory_model_parameters]
        self.memory_model_parameters = ParameterList(memory_model_parameters)

        # torch.func용
        def forward_and_loss(params, x_in, loss_weights, target):
            pred = functional_call(self.memory_model, params, x_in)
            loss = store_memory_loss_fn(pred, target)  # 식 (12)
            weighted_loss = loss * loss_weights
            return weighted_loss.sum(), loss

        self.per_sample_grad_fn = vmap(
            grad(forward_and_loss, has_aux=True),
            in_dims=(0, 0, 0, 0)
        )

        # 입력으로부터 key, value 뽑아내는 projection (식 (11) 참고)
        self.to_keys = nn.Linear(dim, dim * heads, bias=False)
        self.to_values = nn.Linear(dim, dim * heads, bias=False)

        # queries (retrieve)
        self.to_queries = nn.Linear(dim, dim * heads, bias=False)

        self.store_memory_loss_fn = store_memory_loss_fn

        # 모멘텀, 망각(Weight Decay) 파라미터화
        self.momentum = momentum
        self.momentum_order = momentum_order
        self.learned_momentum_combine = learned_momentum_combine

        if momentum:
            # (논문 식 (10)에서 \eta, \theta를 data-dependent 하게)
            self.to_momentum = nn.Linear(dim, heads * momentum_order, bias=True)
        else:
            self.to_momentum = None

        # adaptive learning rate (논문에서 \theta_t)
        self.to_adaptive_step = nn.Linear(dim, heads, bias=True)
        self.adaptive_step_transform = default(
            adaptive_step_transform,
            partial(default_adaptive_step_transform, max_lr=default_step_transform_max_lr)
        )

        # decay factor (논문 식 (13)에서 \alpha_t)
        self.to_decay_factor = nn.Linear(dim, heads, bias=True)

        # AssocScan (직접 보지 못했으나, 내부적으로 prefix-sum 스캔을 가정)
        self.assoc_scan = AssocScan(use_accelerated=False)

        # 파라미터 초기값
        if exists(init_adaptive_step_bias):
            nn.init.constant_(self.to_adaptive_step.bias, init_adaptive_step_bias)
        if exists(init_momentum_bias) and exists(self.to_momentum):
            nn.init.constant_(self.to_momentum.bias, init_momentum_bias)
        if exists(init_decay_bias):
            nn.init.constant_(self.to_decay_factor.bias, init_decay_bias)

    @property
    def memory_model_parameter_dict(self):
        return TensorDict(
            dict(zip(self.memory_model_parameter_names, self.memory_model_parameters)),
            batch_size=[]
        )

    def init_weights(self, batch):
        """초기 메모리 파라미터(M_0) 생성"""
        # [heads, ...] → [b*h, ...]
        weights = self.memory_model_parameter_dict.apply(lambda t: repeat(t, 'h ... -> (b h) ...', b=batch))
        return weights

    def init_momentum(self, batch):
        """초기 모멘텀 S_0 = 0"""
        zeros = self.memory_model_parameter_dict.clone().zero_()
        # [heads, ...] → [momentum_order, b*h, ...]
        zeros = zeros.apply(lambda t: repeat(t, 'h ... -> o (b h) ...', b=batch, o=self.momentum_order))
        return zeros

    def store_memories(
        self,
        seq,
        weights: dict[str, Tensor] | None = None,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        seq_index = 0,
        store_mask: Tensor | None = None,
        return_surprises = True
    ):
        """
        - seq: [batch, n, dim] 형태
        - weights: 현재 메모리 파라미터
        - past_state: (past_update, past_momentum)
        """
        b, n, d = seq.shape
        heads = self.heads

        # chunk 단위 나누기
        chunk_size = self.chunk_size
        n_down = round_down_multiple(n, chunk_size)
        n_rest = n - n_down

        main_seq, remainder = seq[:, :n_down], seq[:, n_down:]
        next_seq_index = seq_index + n_down

        if not exists(weights):
            weights = self.init_weights(b)

        weights_td = TensorDict(weights)


        chunked_seq = main_seq.reshape(b, -1, chunk_size, d)  # [b, #chunks, chunk_size, d]
        chunk_reps = chunked_seq[:, :, 0]                    # [b, #chunks, d]  (첫 토큰만)

        # adaptive lr
        adapt_lr = self.to_adaptive_step(chunk_reps)  # [b, #chunks, heads]
        adapt_lr = self.adaptive_step_transform(adapt_lr)   # sigmoid -> [0, max_lr]

        # forgetting factor (alpha)
        decay = self.to_decay_factor(chunk_reps).sigmoid()  # [b, #chunks, heads]

        # momentum factor (eta)
        if exists(self.to_momentum):
            momentum_factors = torch.sigmoid(self.to_momentum(chunk_reps))  # [b, #chunks, heads*momentum_order]
            momentum_factors = momentum_factors.view(b, -1, heads, self.momentum_order)
        else:
            momentum_factors = None

        # (2) key/value 계산 (식 (11) 참조)
        # ---------------------------------------------------
        # [b, n, dim] --(linear)--> [b, n, heads*dim] --(reshape)--> [b*h, n, dim]
        k = self.to_keys(main_seq)  # [b, n_down, heads*dim]
        v = self.to_values(main_seq)

        k = rearrange(k, 'b n (h d) -> (b h) n d', h=heads)
        v = rearrange(v, 'b n (h d) -> (b h) n d', h=heads)


        chunk_count = main_seq.shape[1] // chunk_size  # #chunks
        adapt_lr = rearrange(adapt_lr, 'b chunk h -> (b h) chunk')
        adapt_lr = adapt_lr.repeat_interleave(chunk_size, dim=1)  # [b*h, chunk * chunk_size]

        # 마스크가 있으면 그 자리는 lr=0
        if exists(store_mask):
            store_mask = store_mask[..., :n_down]  # [b, n_down]
            store_mask = repeat(store_mask, 'b n -> (b h) n', h=heads)
            adapt_lr = torch.where(store_mask, adapt_lr, adapt_lr.new_zeros(()))

        # per-sample grad 계산
        # grads: TensorDict
        grads, raw_loss = self.per_sample_grad_fn(
            dict(weights_td),  # 현재 메모리 파라미터
            k,                 # 모델 입력 (keys)
            adapt_lr,          # loss_weights
            v                  # target (values)
        )
        grads = TensorDict(grads)
        raw_loss = raw_loss.reshape(-1, 1)  # 그냥 shape 보정

        # (4) surprise = - grad  (식 (10)에서 - θ_t ∇ℓ)
        # ---------------------------------------------------
        surprises = grads.apply(lambda t: -t)

        # (5) 모멘텀 / 망각(식 (13)) 업데이트
        # ---------------------------------------------------
        if not exists(past_state):
            # (past_update, past_momentum) 둘 다 0 초기화
            init_up = weights_td.clone().zero_()
            init_mo = self.init_momentum(b)
            past_state = (init_up, init_mo)

        past_last_update, past_last_momentum = past_state

        # 아래 두 텐서 사전으로 최종 업데이트 결과를 저장
        updates = TensorDict()
        next_last_update = {}
        next_last_momentum = {}


        # ---------------------------------------------------
        chunked_surprises = rearrange_dict_values(surprises, '(b h) n ... -> (b h) chunk_size n_chunk ...', chunk_size=chunk_size, n_chunk=chunk_count)

        # momentum + forgetting scan
        for param_name, chunk_val in chunked_surprises.items():

            last_mo = past_last_momentum[param_name]  # shape: [momentum_order, b*h, ...] 라고 가정
            last_up = past_last_update[param_name]    # shape: [b*h, ...]

            # 최종적으로 모든 chunk step의 업데이트 결과를 concat할 리스트
            all_updates = []

            current_mo = last_mo[-1]

            # scan하며 누적할 마지막 모멘텀
            new_momentums = []


            for c_idx in range(chunk_count):
                # c_idx번째 chunk의 surprise
                # [b*h, chunk_size, ...]
                chunk_surprise = chunk_val[:, :, c_idx]


                eta_t = momentum_factors[:, c_idx] if exists(momentum_factors) else None
                alpha_t = decay[:, c_idx]          # [b, chunk, heads] -> [ (b h), ] 변환 필요
 
                if exists(eta_t):

                    current_mo = self.assoc_scan(eta_t, chunk_surprise, prev=current_mo, remove_prev=True)
                else:
                    # momentum이 없는 경우 그냥 surprise = S_t
                    current_mo = chunk_surprise

                updated_val = (1. - alpha_t) * last_up + current_mo

                # update를 리스트에 쌓기
                all_updates.append(updated_val)

                last_up = updated_val

            param_updates = torch.stack(all_updates, dim=1)

            updates[param_name] = param_updates
            next_last_update[param_name] = param_updates[:, -1]  # 마지막 chunk 스텝의 값
            # 모멘텀도 마찬가지
            if exists(self.to_momentum):
                # 여러 order가 있다면 여기서 갱신
                next_last_momentum[param_name] = torch.stack([current_mo], dim=0)
            else:
                next_last_momentum[param_name] = past_last_momentum[param_name]

        # state 갱신
        new_state = (next_last_update, next_last_momentum)
        next_store_state = NeuralMemState(
            next_seq_index,
            weights_td,           # 아직은 "기본 weights" (각 chunk별로는 업데이트 누적한 스냅샷이 updates에 들어있음)
            remainder,
            new_state,
            updates
        )

        if not return_surprises:
            return updates, next_store_state
        # raw_loss, adapt_lr 등 필요하시면 함께 리턴
        return updates, next_store_state, raw_loss

    def retrieve_memories(self, seq, updates: dict[str, Tensor]):
        """
        (논문 3.1 마지막 부분)
        query를 통해 메모리(파라미터)에 forward pass -> value 얻기
        """
        b, n, d = seq.shape
        heads = self.heads

        # query
        q = self.to_queries(seq)  # [b, n, heads*dim]
        q = rearrange(q, 'b n (h d) -> (b h) n d', h=heads)


        final_weights = {}
        for param_name, upd in updates.items():
            final_weights[param_name] = upd[:, -1]  # 마지막 chunk 스텝

        final_weights_td = TensorDict(final_weights, batch_size=[])

        # functional_call 로 retrieve
        v_out = functional_call(self.memory_model, final_weights_td, q)  # [ (b h), n, d ]
        # [b, h, n, d]
        v_out = rearrange(v_out, '(b h) n d -> b h n d', b=b, h=heads)
        # 원하는 식으로 head 병합
        v_out = rearrange(v_out, 'b h n d -> b n (h d)')

        return v_out

    def forward(
        self,
        seq,
        state: NeuralMemState | None = None,
        store_mask: Tensor | None = None,
        detach_mem_state=False,
        return_surprises=False
    ):
        """
        seq: [b, n, d]
        state: (seq_index, weights, cache_store_seg, (past_update, past_momentum), updates)
        """
        if not exists(state):
            state = NeuralMemState(0, None, None, None, None)

        seq_index, weights, cache_store_seg, past_state, updates = state

        if exists(cache_store_seg):
            seq = safe_cat((cache_store_seg, seq), dim=-2)

        next_updates, next_mem_state, raw_loss = self.store_memories(
            seq,
            weights,
            past_state,
            seq_index=seq_index,
            store_mask=store_mask,
            return_surprises=True
        )

        # 2) retrieve
        out = self.retrieve_memories(seq, next_updates)

        if detach_mem_state:
            next_mem_state = mem_state_detach(next_mem_state)

        if not return_surprises:
            return out, next_mem_state

        return out, next_mem_state, raw_loss
