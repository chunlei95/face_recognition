from utils.logger import setup_logger


def evaluate(model,
             val_loader,
             log_dir):
    logger = setup_logger(name='eval', output=log_dir)
    logger.info('start evaluate...')
    for idx, data in enumerate(val_loader):
        output = model(data)
        calculate_map(output, data['target'])
    logger.info(f'The evaluate mAP = {0}, '
                f'time cost = {0}, '
                f'avg inference time = {0}ms, '
                f'FPS = {0}images/s')
    logger.info('end evaluate...')


def calculate_map(predict, target):
    pass
