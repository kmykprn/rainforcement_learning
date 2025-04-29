import torch


def test(device, dataloader, model, loss_func):

    # 評価モードに設定
    model.eval()

    datasize = len(dataloader.dataset)

    # lossと正解率を評価する変数
    test_loss, correct = 0, 0

    # テストなので勾配の保存をキャンセル
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            # 計算をGPU上で行なう
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            test_loss += loss_func(pred, y_batch).item()
            correct += (pred.argmax(1) == y_batch).type(torch.float).sum().item()

    test_loss /= datasize
    correct /= datasize
    print(
        f"Test Error:\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
