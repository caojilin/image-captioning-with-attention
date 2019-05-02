from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr8k',
                       karpathy_json_path='split/dataset_flickr8k.json',
                       image_folder='datasets_8k/Flicker8k_Dataset',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='datasets_8k',
                       max_len=50)
