def classifier_train_step(batch_size, optimizer, criterion, images, labels):
    optimizer.zero_grad()
    outputs = classifieur(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()