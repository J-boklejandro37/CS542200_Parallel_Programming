import argparse
import math
import json
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import flash_attn_qkvpacked_func
from flash_attn.utils.benchmark import benchmark_fwd_bwd

def flops(batch_size, seq_len, head_dim, num_heads, causal, mode='fwd'):
    assert mode in ['fwd', 'bwd', 'fwd_bwd']
    f = 4 * batch_size * seq_len**2 * num_heads * head_dim // (2 if causal else 1)
    return f if mode == 'fwd' else (2.5 * f if mode == 'bwd' else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean

def pytorch_attn_func(qkv, dropout_p=0.0, causal=True):
    batch_size, seq_len, _, num_heads, head_dim = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(head_dim)

    scores = torch.empty(batch_size * num_heads, seq_len, seq_len, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=num_heads)
    if causal:
        causal_mask = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=scores.device), 1)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)

def benchmark_attention(batch_size, seq_len, num_heads, emb_dim, impl, causal, repeats, output):
    assert impl in ['Pytorch', 'Flash2']
    device = 'cuda'
    dtype = torch.float16
    dropout_p = 0.0
    head_dim = emb_dim // num_heads
    attention_func = flash_attn_qkvpacked_func if impl == 'Flash2' else pytorch_attn_func

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    qkv = torch.randn(
        batch_size, seq_len, 3, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True, 
    )
    forward_time, backward_time = time_fwd_bwd(
        attention_func, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False, 
    )
    peak_memory_usage = torch.cuda.max_memory_allocated() / (1024**2)

    benchmark_result = {
        'forward': {
            'time(s)': forward_time, 
            'FLOPS(TFLOPs/s)': efficiency(
                flops(batch_size, seq_len, head_dim, num_heads, causal, mode='fwd'), forward_time
            )
        }, 
        'backward': {
            'time(s)': backward_time, 
            'FLOPS(TFLOPs/s)': efficiency(
                flops(batch_size, seq_len, head_dim, num_heads, causal, mode='bwd'), backward_time
            )
        }, 
        'forward_backward': {
            'time(s)': forward_time + backward_time, 
            'FLOPS(TFLOPs/s)': efficiency(
                flops(batch_size, seq_len, head_dim, num_heads, causal, mode='fwd_bwd'), forward_time + backward_time
            )
        }, 
        'peak_memory_usage(MB)': peak_memory_usage, 
    }

    with open(output, 'w') as json_file:
        json.dump(benchmark_result, json_file, indent=2)

def validate_args(args):
    if args.emb_dim % args.num_heads != 0:
        raise ValueError('--num_heads must be divisible by --emb_dim')

    if args.impl not in ['Pytorch', 'Flash2']:
        raise ValueError("--impl must be one of ['Pytorch', 'Flash2']")
    
    if args.output[-5:] != '.json':
        raise ValueError("--output must be a filename ending with '.json'")

    return args

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate the FLOPS and memory consumption of the attention mechanism.')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training. Default is 32.'
    )

    parser.add_argument(
        '--seq_len',
        type=int,
        default=1024,
        help='Sequence length. Default is 1024.'
    )

    parser.add_argument(
        '--num_heads',
        type=int,
        default=32,
        help='Number of attention heads. Must be divisible by emb_dim. Default is 32.'
    )

    parser.add_argument(
        '--emb_dim',
        type=int,
        default=2048,
        help='Embedding dimension. Default is 2048.'
    )

    parser.add_argument(
        '--impl',
        type=str,
        default='Flash2',
        choices=['Pytorch', 'Flash2'],
        help="Implementation type. Must be one of ['Pytorch', 'Flash2']. Default is 'Flash2'."
    )

    parser.add_argument(
        '--causal',
        action='store_true',
        help='If set, enables causal attention. Default is False.'
    )

    parser.add_argument(
        '--repeats',
        type=int,
        default=30,
        help='Number of repeats for evaluation. Default is 30.'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_result.json',
        help='The JSON filename for the benchmark result output file. Default is benchmark_result.json.'
    )

    args = parser.parse_args()
    return validate_args(args)

def main():
    args = parse_arguments()
    benchmark_attention(**vars(args))

if __name__ == '__main__':
    main()

