"""CPU-only, explicit request-vs-token denominators for the timing benchmark."""
import math
import statistics


def request_metrics(*, ttft_seconds, decode_seconds, generated_lengths, decode_steps):
    times = (ttft_seconds, decode_seconds)
    if not all(math.isfinite(t) for t in times) or ttft_seconds <= 0 or decode_seconds < 0:
        raise ValueError('invalid measurement duration')
    if not generated_lengths or any(type(n) is not int or n < 1 for n in generated_lengths):
        raise ValueError('each request must generate at least the first token')
    if decode_steps != max(generated_lengths)-1 or (decode_steps == 0) != (decode_seconds == 0):
        raise ValueError('decode steps/time inconsistent with generated lengths')
    generated = sum(generated_lengths)
    decode_generated = generated-len(generated_lengths)
    elapsed = ttft_seconds+decode_seconds
    return dict(batch_size=len(generated_lengths), generated_lengths=generated_lengths,
        generated_tokens=generated, decode_generated_tokens=decode_generated,
        decode_steps=decode_steps, ttft_seconds=ttft_seconds, decode_seconds=decode_seconds,
        elapsed_seconds=elapsed, e2e_tokens_per_second=generated/elapsed,
        decode_tokens_per_second=decode_generated/decode_seconds if decode_seconds else None,
        decode_ms_per_step=1000*decode_seconds/decode_steps if decode_steps else None,
        decode_ms_per_output_token=1000*decode_seconds/decode_generated if decode_generated else None)


def percentile(values, q):
    ordered = sorted(values)
    p = (len(ordered)-1)*q
    lower, upper = math.floor(p), math.ceil(p)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(p-lower)


def describe(values):
    med = statistics.median(values)
    return dict(n=len(values), median=med, q25=percentile(values,.25), q75=percentile(values,.75),
        iqr=percentile(values,.75)-percentile(values,.25),
        mad=statistics.median(abs(v-med) for v in values), min=min(values), max=max(values))


def aggregate(rows):
    if not rows:
        raise ValueError('empty measurements')
    result = dict(request_count=len(rows))
    for key in ('ttft_seconds','decode_seconds','elapsed_seconds','e2e_tokens_per_second',
                'decode_tokens_per_second','decode_ms_per_step','decode_ms_per_output_token',
                'peak_allocated_bytes','peak_reserved_bytes','resident_allocated_bytes',
                'resident_reserved_bytes','tokenization_seconds','text_e2e_seconds',
                'cuda_ttft_seconds','cuda_decode_seconds'):
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:result[key] = describe(values)
    decode_seconds = math.fsum(row['decode_seconds'] for row in rows)
    result['pooled_e2e_tokens_per_second'] = sum(row['generated_tokens'] for row in rows)/math.fsum(row['elapsed_seconds'] for row in rows)
    result['pooled_decode_tokens_per_second'] = sum(row['decode_generated_tokens'] for row in rows)/decode_seconds if decode_seconds else None
    return result
