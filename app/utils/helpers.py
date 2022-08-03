def bool2str(v):
    bn = 1 if str(v).lower() in ["yes", "true", "t", "1"] else 0
    return str(bn)