
from lib.data import get_data, prepare_img_tensor
from lib.utils import show_batch
import matplotlib.pyplot as plt

def print_data_dict(d : dict):
    line_str = '-'*100
    for k, v in d.items():
        print(k)
        print(v)
        print(line_str)

if __name__ == '__main__':
    
    data_dict = get_data()

    train_dataset, test_dataset = data_dict['train_dataset'], data_dict['test_dataset']
    
    train_dataloader, test_dataloader = train_dataset.to_dataloader(batch_size=16, shuffle=True), test_dataset.to_dataloader(batch_size=16, shuffle=True)
    
    '''
    de acordo com esse print, parece que as imagens estão ok
    X, y, img = next(iter(train_dataloader))
    for img_t in img:
        img_t = prepare_img_tensor(img_t, [0.5]*3, [0.5]*3)
        plt.imshow(img_t)
        plt.show()
    '''