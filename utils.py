import logging
import multiprocessing
import os
import Queue
import sys
import threading
import time
import urllib

import numpy as np

from fivehundredpx.client import FiveHundredPXAPI


def load_data(consumer_key):
    """
    This function uses consumer key and return a dictionary of photo information in 5 different
    ranges
    """
    api = FiveHundredPXAPI()
    data_set_dict = {'80-100': [], '60-80': [], '40-60': [], '20-40': [], '0-20': []}
    for page_num in range(1506, 2000):
        photos = api.photos(consumer_key=consumer_key,
                    feature='fresh_week',
                    image_size=21,
                    sort_direction='asc',
                    page=page_num,
                    rpp=50)
        for photo in photos['photos']:
            highest_rating = photo['highest_rating']
            logging.info("The hightst rating is {}".format(highest_rating))
            if 80.0 <= highest_rating <= 100.0:
                data_set_dict['80-100'].append(photo)
            elif 60.0 <= highest_rating < 80.0:
                data_set_dict['60-80'].append(photo)
            elif 40.0 <= highest_rating < 60.0:
                data_set_dict['40-60'].append(photo)
            elif 20.0 <= highest_rating < 40.0:
                data_set_dict['20-40'].append(photo)
            elif 0.0 < highest_rating < 20.0:
                data_set_dict['0-20'].append(photo)
            elif highest_rating == 0.0:
                if photo['votes_count'] > 0:
                    data_set_dict['0-20'].append(photo)
            else:
                logging.warning("{} rate is not a valid rate".format(highest_rating))

    logging.warning("interval:80-100, image_num:{}".format(len(data_set_dict['80-100'])))
    logging.warning("interval:60-80, image_num:{}".format(len(data_set_dict['60-80'])))
    logging.warning("interval:40-60, image_num:{}".format(len(data_set_dict['40-60'])))
    logging.warning("interval:20-40, image_num:{}".format(len(data_set_dict['20-40'])))
    logging.warning("interval:0-20, image_num:{}".format(len(data_set_dict['0-20'])))

    return data_set_dict

def reduce_data_set(data_set_dict, each_group_size=2000):
    data_set_dict_reduced = {'80-100': [], '60-80': [], '40-60': [], '20-40': [], '0-20': []}
    data_set_dict_reduced['80-100'] = np.asarray(data_set_dict['80-100'])[np.random.choice(len(data_set_dict['80-100']), each_group_size)]
    data_set_dict_reduced['60-80'] = np.asarray(data_set_dict['60-80'])[np.random.choice(len(data_set_dict['60-80']), each_group_size)]
    data_set_dict_reduced['40-60'] = np.asarray(data_set_dict['40-60'])[np.random.choice(len(data_set_dict['40-60']), each_group_size)]
    data_set_dict_reduced['20-40'] = np.asarray(data_set_dict['20-40'])[np.random.choice(len(data_set_dict['20-40']), each_group_size)]
    data_set_dict_reduced['0-20'] = np.asarray(data_set_dict['0-20'])
    return data_set_dict_reduced

def create_list_images(data_set, main_dir):
    img_list = []
    for key, pics in data_set.iteritems():
        dir_path = os.path.join(main_dir, key)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for pic in pics:
            img_url = pic['image_url']
            img_name = ''.join ([img_url.split('/')[-1], '.jpg'])
            img_full_path = os.path.join(dir_path, img_name)
            img_list.append((img_url, img_full_path))
    return img_list


def thread_worker(q, exit_signal):
    """ Worker to download images"""
    sys.stdout.flush()
    while not exit_signal.is_set():
        try:
            url, img_path = q.get_nowait()
        except Queue.Empty:
            time.sleep(1)  # Prevent worker from hammering CPU
            continue

        try:
            urllib.urlretrieve(url, img_path)
        except:
            logging.exception("Failed to download %s to %s", url, img_path)

        q.task_done()
    sys.stdout.flush()


def download_files_from_urls(url_path_list):
    """
    Download files from list of URLs to output directory.

    url_path_list - list of tuples of a URL and a download destination
    """
    queue = Queue.Queue()
    process_count = min(multiprocessing.cpu_count() * 10, len(url_path_list))
    exit_signal = threading.Event()

    output = []
    try:
        # Create executors
        logging.debug("Creating {} workers to download files".format(process_count))
        for i in range(process_count):
            worker = threading.Thread(
                target=thread_worker,
                args=(queue, exit_signal)
            )
            worker.daemon = True
            worker.start()

        logging.info("Downloading {} files".format(len(url_path_list)))
        for url, file_path in url_path_list:
            queue.put((url, file_path))
            output.append(file_path)

        # Hangout, wait for tasks to complete
        queue.join()
        logging.info("Successfully downloaded {} files".format(len(url_path_list)))
        return output

    finally:
        logging.debug("Shutting down workers")
        exit_signal.set()

def write_train_test_files(data_set_path, text_file_name):
    label_map = data_set_dict_reduced = {'80-100': 0, '60-80': 1, '40-60': 2, '20-40': 3, '0-20': 4}

    for root, dirs, files in os.walk(data_set_path):
        for file_each in files:
            if file_each.endswith('.jpg'):
                training_txt = "{0} {1}\n".format(os.path.join(root, file_each), label_map[root.split('/')[-1]])
                testing_txt = "{0} {1}\n".format(os.path.join(root, file_each), label_map[root.split('/')[-1]])
                with open(test_file_name, "a") as train_file:
                    train_file.write(training_txt)


def download_imgs(data_set_all=None, main_dir = 'dataset'):
    if data_set_all is None:
        with open('json_report.json', 'r') as result_report:
            data_set_all = json.load(result_report)

    data_set_reduced = reduce_data_set(data_set_all)
    url_img_path = create_list_images(data_set_reduced, main_dir)
    downloaded_files = download_files_from_urls(url_img_path)



def read_txt_list(data_path):
    data_list = {}
    with open(data_path) as data_file:
        lines = data_file.readlines()
        for l in lines:
            items = l.split()
            data_list.setdefault(str(items[1]),[]).append(items[0])
    return data_list


def get_class_stat(data_list):
    data_stats = {"class_name":["0", "1", "2", "3", "4"], "num_data":[]}
    for each_class in data_stats["class_name"]:
        data_stats["num_data"].append(len(data_list[each_class]))
    return data_stats
