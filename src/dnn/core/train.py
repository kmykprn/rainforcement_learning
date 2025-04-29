def train(device, dataloader, model, loss_func, optimizer):
    """
    深層学習の学習用関数
    """

    # 訓練モードに設定。BatchNormやDropoutに効く。
    model.train()

    datasize = len(dataloader.dataset)

    for batch, (x_batch, y_batch) in enumerate(dataloader):

        # 計算をGPU上で行なう
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # 前回の勾配をリセット
        optimizer.zero_grad()

        # 入力データに対して予測を実行
        pred_batch = model(x_batch)

        # lossを計算
        loss = loss_func(pred_batch, y_batch)

        # 勾配を計算
        loss.backward()

        # 重みを更新
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x_batch)
            print(f"loss: {loss:.7f}  [{current:5d}/{datasize:5d}]")
