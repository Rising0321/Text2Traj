import os


def calc_files(walk_path):
    res = []
    for root, nowdir, nowfiles in os.walk(walk_path):
        # print(nowfiles)
        for file in nowfiles:
            if file.find("data") != -1:
                res.append(root + "/" + file)
    return res


def myprepare(accelerator, model, optimizer, lr_scheduler):
    model = accelerator.prepare_model(model)
    optimizer = accelerator.prepare_optimizer(optimizer)
    lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)
    return model, optimizer, lr_scheduler
