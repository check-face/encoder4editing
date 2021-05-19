import argparse

import torch
import numpy as np
import sys
import os
import dlib
import PIL
import threading
import flask
import queue
import io

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from utils.model_utils import setup_model
from utils.alignment import align_face, NumberOfFacesError


dlib_shape_predictor = dlib.shape_predictor(paths_config.model_paths['shape_predictor'])

PIL.Image.MAX_IMAGE_PIXELS = 8192 * 4096 # protect against decompression bomb DOS attacks

app = flask.Flask(__name__)

job_queue = queue.Queue()


class EncodeJob:
    def __init__(self, img, name, align):
        self.img = img
        self.align = align
        self.name = name
        self.evt = threading.Event()
        self.did_align = False

    def __str__(self):
        return self.name

    def set_result(self, latent):
        self.latent = latent
        self.evt.set()

    def wait_for_result(self, timeout):
        if self.evt.wait(timeout):
            return self.latent
        else:
            return None

@app.route('/api/encodeimage/', methods=['POST'])
def encode_image():

    tryalign = flask.request.form.get('tryalign', 'false')
    do_align = tryalign.lower() == 'true'
    file = flask.request.files['usrimg']
    if not file:
        return flask.Response('No file uploaded for usrimg', status=400)

    img = PIL.Image.open(file.stream)
    img = img.convert("RGB") # processing steps are expecting an RGB image
    job = EncodeJob(img, 'user uploaded file', do_align)
    job_queue.put(job)
    latent = job.wait_for_result(15)
    if latent is None:
        raise Exception("Encoding image failed or timed out")
    did_align = job.did_align == True

    return flask.jsonify({'dlatent': latent.tolist(), 'did_align': did_align})

def run_alignment(img):
    try:
        aligned_image = align_face(filepath=None, predictor=dlib_shape_predictor, img=img)
        print("Aligned image has shape: {}".format(aligned_image.size))
        return aligned_image
    except NumberOfFacesError:
        return None

def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(net, batches, is_cars=False):
    all_latents = []
    with torch.no_grad():
        for batch in batches:
            x = batch
            inputs = x.to(device).float()
            latents = get_latents(net, inputs, is_cars)
            all_latents.append(latents)
    return torch.cat(all_latents)

def get_batch(batchsize):
    yield job_queue.get(True) # will block until it gets a job
    for i in range(batchsize-1):
        if not job_queue.empty():
            yield job_queue.get_nowait()

def worker(ckpt, batch_size):
    net, opts = setup_model(ckpt, device)
    is_cars = 'car' in opts.dataset_type
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    transform = transforms_dict['transform_inference']

    print("Worker ready")

    while True:
        jobs = list(get_batch(batch_size))
        app.logger.info(f"Running jobs {[str(job) for job in jobs]}")

        imgs = []
        for job in jobs:
            if job.align:
                aligned = run_alignment(job.img)
                if aligned is None:
                    imgs.append(job.img)
                else:
                    imgs.append(aligned)
                    job.did_align = True
            else:
                imgs.append(job.img)                

        batch = torch.stack([transform(img) for img in imgs])
        latents = get_all_latents(net, [batch], is_cars=is_cars)
        latents = latents.cpu()

        for latent, job in zip(latents, jobs):
            job.set_result(np.array(latent))

        app.logger.info("Finished batch job")

def main(args):
    t1 = threading.Thread(target=worker, args=[args.ckpt, args.batch])
    t1.daemon = True # kill thread on program termination (to allow keyboard interrupt)
    t1.start()

    app.config['TEMPLATES_AUTO_RELOAD'] = True # for debugging
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # limit to 16mb
    app.run(host="0.0.0.0", port=args.api_port)
    app.logger.info("Closing server_inference")

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Run a flask server to do e4e inference on images")
    parser.add_argument("--batch", type=int, default=1, metavar="BATCH_SIZE", help="batch size for the generator")
    parser.add_argument("--api-port", type=int, default=8080, metavar="PORT", help="port to listen on (default 8080)")
    parser.add_argument("--ckpt", default="pretrained_models/e4e_ffhq_encode.pt", metavar="CHECKPOINT", help="path to generator checkpoint")

    args = parser.parse_args()
    main(args)
