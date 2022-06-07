def golden_cross(fast, last_fast, slow, last_slow):
    return (last_fast < fast) and (fast > slow)


def death_cross(fast, last_fast, slow, last_slow):
    return (last_fast > fast) and (fast < slow)


def ma_up_penetrate(price, ma, last_ma):
    return (price > ma) and (last_ma < ma)


def ma_down_penetrate(price, ma, last_ma):
    return (price < ma) and (last_ma > ma)


def constant_compare(idx, op, constant):
    return eval(str(idx) + op + str(constant), locals())


def bounce_back_up(last_price, c_price, last_idx, c_idx):
    return (last_price <= last_idx) and (c_price >= c_idx)


def bounce_back_down(last_price, c_price, last_idx, c_idx):
    return (last_price >= last_idx) and (c_price <= c_idx)
