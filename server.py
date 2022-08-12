import io 
import zmq 
import zmq.asyncio as aiozrmq 

import cv2
import numpy as np 
import itertools as it, functools as ft 

import json 
import pickle 

from libraries.log import logger 

from fastapi import FastAPI, UploadFile, File 
from fastapi.responses import JSONResponse

from constants import ZMQ_SERVER_PORT

app = FastAPI()
ctl = FastAPI(
    title="deep learning image classification", 
    description="""
        This app is an image classification tool
        It is based on pytorch, zeromq and fastapi 
        You can use /backend/inference to make prediction  
    """
)

ctx = None 
dealer_socket = None 

@app.on_event('startup')
async def handle_startup():
    global ctx 
    global dealer_socket

    ctx = aiozrmq.Context()
    dealer_socket = ctx.socket(zmq.DEALER)
    dealer_socket.setsockopt(zmq.LINGER, 0)
    dealer_socket.connect(f'tcp://localhost:{ZMQ_SERVER_PORT}')

    logger.success('api service is up and ready the exchange messages')

@app.on_event('shutdown')
async def handle_shutdown():
    dealer_socket.close()
    ctx.term() 
    logger.success('api service has removed all ressources')

@app.get('/')
async def handle_entrypoint():
    return JSONResponse(
        status_code=200, 
        content=f'deep learning image classification => go to /backend/docs to use the api'
    )

@ctl.get('/')
async def handle_entrypoint():
    return JSONResponse(
        status_code=200, 
        content=f'the serice is up and ready to make prediciton. => /docs'
    )

@ctl.post('/inference')
async def handle_infenrece(incoming_image:UploadFile=File(...)):
    try:
        image_bytestream = await incoming_image.read()
        image_vector = np.frombuffer(image_bytestream, dtype=np.uint8)
        image = cv2.imdecode(image_vector, cv2.IMREAD_COLOR)

        await dealer_socket.send_multipart([b''], flags=zmq.SNDMORE)
        await dealer_socket.send_pyobj(image)

        _, encoded_response = await dealer_socket.recv_multipart()
        decoded_response = json.loads(encoded_response)
        return JSONResponse(**decoded_response)
    except Exception as e:
        catched_message = f'Exception : {e}'
        return JSONResponse(
            status_code=400, 
            content=catched_message
        )
