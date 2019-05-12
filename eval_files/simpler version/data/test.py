import json

with open("dataset_coco.json", 'rb') as f:
    data = json.load(f)

images = data["images"]

def get_caption_from_image(file_name):
    def get_image_info(file_name):
        for im in images:
            if im["filename"] == file_name:
                return im
        raise Exception("image not found")

    image = get_image_info(file_name)
    captions = []
    sentences = image["sentences"]
    for i in sentences:
        raw = i["raw"]
        raw = raw.rstrip("\n")
        captions.append(raw)
    return captions[0:5]


lineList = [line.rstrip('\n') for line in open("filename.txt")]


def get_kth_caption(filenames):
    for file in filenames:
        caption = get_caption_from_image(file)
        if len(caption) != 5:
            print(caption)
        for index, elem in enumerate(caption):
            with open('eval_files/ref{0}.txt'.format(index + 1), 'a') as the_file:
                the_file.write('{0}\n'.format(elem))

