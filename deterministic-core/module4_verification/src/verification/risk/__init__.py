"""Risk-certified verification (Check B): calibrated bands, scenarios and the certificate.

Data-calibrated, not learned: the bands are counted from past persistence misses offline and
frozen in a versioned artifact; at runtime there is a table lookup, seeded sampling, power flows
and an exact binomial bound. No ML import anywhere (ADR 0003).
"""
