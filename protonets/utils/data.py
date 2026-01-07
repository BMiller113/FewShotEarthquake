import protonets.data


def load(opt, splits):
    ds_name = opt["data.dataset"]

    if ds_name == "omniglot":
        ds = protonets.data.omniglot.load(opt, splits)
    elif ds_name in ["xbd", "xbd_eq", "xview2", "xbd_xview2"]:
        ds = protonets.data.xbd.load(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(ds_name))

    return ds
