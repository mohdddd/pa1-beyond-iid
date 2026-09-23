def build_method(cfg):
    name = cfg["method"]
    if name == "vanilla":
        from task4.methods.vanilla import Vanilla as M
    elif name == "gcsc":
        from task4.methods.gcsc import GCSC as M
    elif name == "proser":
        from task4.methods.proser import PROSER as M
    else:
        raise ValueError(name)
    return M(cfg)
