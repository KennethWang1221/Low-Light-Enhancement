#!/usr/bin/env python3

import sys
import argparse
import cv2
import os
import numpy as np
import glob

DEBUG = False

def frames_to_video(frames_folder, output_video_path, fps):
    frame_files = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith('.png')])
    if not frame_files:
        raise ValueError("No frames found in the specified folder.")
    
    # Read the first frame to get the dimensions
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video.write(frame)

    video.release()

def build_onnx_model(onnx_file):
    providers = ["CPUExecutionProvider"]
    import onnxruntime

    session = onnxruntime.InferenceSession(onnx_file,providers=providers)
    session.get_modelmeta()
    meta = session.get_modelmeta().custom_metadata_map  # metadata
    return session

def zero_dce_inference(model, im, file_name, inference_input_tensor_path, inference_opt_tensor_path):
    model_output_details = model.get_inputs()[0].name
    input_details = model.get_outputs()[0].name
    
    if DEBUG:
        np.savetxt('{}/{}_input.tensor'.format(inference_input_tensor_path, file_name),
            im.reshape(-1), fmt='%.8f')

    output = model.run(None, {
        model_output_details: im})[0]
    
    if DEBUG:
        np.savetxt('{}/{}_output.tensor'.format(inference_opt_tensor_path, file_name),
                output.reshape(-1), fmt='%.8f')

    return output

def preprocess(im, file_path):
    file_name =  os.path.splitext(os.path.basename(file_path))[0] 
    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_ori = im.copy()

    im = im / 255.    
    im_ori = im_ori / 255.
    
    im = cv2.resize(im, (160, 90),interpolation=cv2.INTER_LINEAR)
    im = np.transpose(im,(2,0,1)).astype(np.float32)
    im_ori = np.transpose(im_ori,(2,0,1)).astype(np.float32)
    
    im = np.expand_dims(im,axis=0)
    im_ori = np.expand_dims(im_ori,axis=0)

    return im, im_ori, file_name

def zero_dce_postprocess(inference_res, im_ori, file_name, image_results_path):

    inference_res = np.squeeze(inference_res,axis=0)
    inference_res = inference_res.transpose(1,2,0)
    inference_res = np.tanh(inference_res) 
    inference_res = cv2.resize(inference_res, (im_ori.shape[3], im_ori.shape[2]), interpolation=cv2.INTER_LINEAR)
    inference_res = inference_res.transpose(2,0,1)
    
    x_r = np.expand_dims(inference_res,axis=0)
    x = im_ori.copy()

    x = x + x_r*(np.power(x,2)-x)
    x = x + x_r*(np.power(x,2)-x)
    x = x + x_r*(np.power(x,2)-x)
    x = x + x_r*(np.power(x,2)-x)		
    x = x + x_r*(np.power(x,2)-x)				
    x = x + x_r*(np.power(x,2)-x)	
    x = x + x_r*(np.power(x,2)-x)
    enhance_image = x + x_r*(np.power(x,2)-x)

    opt = enhance_image * 255
    opt = np.clip(opt, 0, 255)
    opt = np.squeeze(opt,axis=0)
    opt = opt.transpose(1,2,0).astype(np.uint8)
    opt = cv2.cvtColor(opt, cv2.COLOR_RGB2BGR)
    
    image_path = os.path.join(image_results_path, '{}.png'.format(file_name))
    cv2.imwrite(image_path, opt) 

def main(**args):
    if os.access(args['model_file'], os.R_OK) == 0:
        print('cannot access network binary {}'.format(args['model_file']))
        sys.exit(1)

    if not os.path.exists(args['opts_dir']):
        os.makedirs(args['opts_dir'])

    video_frames = os.path.join(args['opts_dir'], 'video_frames')
    inference_opt_tensor_path = os.path.join(args['opts_dir'], 'inference_opt_tensor')
    image_results_path = os.path.join(args['opts_dir'], 'image_res') 
    inference_input_tensor_path = os.path.join(args['opts_dir'], 'inference_input_tensor')
    preprocess_res_path = os.path.join(args['opts_dir'], 'preprocess_res')
    mp4_res = os.path.join(args['opts_dir'], '{}.mp4'.format(os.path.basename(args['opts_dir'])))

    if not os.path.exists(inference_opt_tensor_path):
        os.makedirs(inference_opt_tensor_path)
    if not os.path.exists(image_results_path):
        os.makedirs(image_results_path)
    if not os.path.exists(inference_input_tensor_path):
        os.makedirs(inference_input_tensor_path)
    if not os.path.exists(preprocess_res_path):
        os.makedirs(preprocess_res_path)
    if not os.path.exists(video_frames):
        os.makedirs(video_frames)

    model = build_onnx_model(args['model_file'])

    cap = cv2.VideoCapture(args['video_path'])
    frame_count = 0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(video_frames, f'frame_{frame_count:04d}.png')
        if DEBUG:
            cv2.imwrite(frame_filename, frame)
    
        im, im_ori, file_name = preprocess(frame, frame_filename)

        inference_res = zero_dce_inference(model, im, file_name, inference_input_tensor_path, inference_opt_tensor_path)

        zero_dce_postprocess(inference_res, im_ori, file_name, image_results_path)

        frame_count += 1

    cap.release()
    frames_to_video(image_results_path, mp4_res, fps=int(frame_rate))


if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    # Load file
    parser.add_argument("--model_file", type=str,default="./models/zero_dce.onnx", \
                        help='path to model')
    parser.add_argument("--video_path", type=str, default="./demo.mp4", \
                        help='path to image folder')
    # Dirs
    parser.add_argument("--opts_dir", type=str, default="./res", \
                        help='path of outputs files ')
    argspar = parser.parse_args()    

    print("\n### Test model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
