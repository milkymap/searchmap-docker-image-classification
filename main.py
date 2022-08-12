import zmq 
import click 
import pickle 
import uvicorn

import numpy as np 
import torch.multiprocessing as mp 

from os import path, getenv  
from PIL import Image 
from rich.progress import Progress

from torch.utils.data import TensorDataset, DataLoader

from libraries.log import logger
from libraries.strategies import *

from constants import ZMQ_SERVER_PORT, MAP_BACKBONE2NB_FEATURES
from specializer import MLPModel
from server import app, ctl 

@click.group(chain=True, invoke_without_command=True)
@click.option('--backbone', help='backbone (model_name)', default='resnet18')
@click.option('--img_extension', help='image file extension', default='*.jpg')
@click.option('--task_labels_config', help='task labels config', default='labels_config.csv')
@click.option('--embeddings_filename', help='name of the file that hold extracted features', default='embeddings.pkl')
@click.pass_context
def router_command(ctx, backbone, img_extension, task_labels_config, embeddings_filename):
    ctx.ensure_object(dict)
    subcommand = ctx.invoked_subcommand 
    if subcommand is None: 
        logger.debug('use --help option in order to see the available subcommands')
    else:
        logger.debug(f'{subcommand} was called')
        
        path2models = getenv('MODELS')
        path2dataset = getenv('DATASET')
        path2features = getenv('FEATURES')
        path2checkpoints = getenv('CHECKPOINTS')
        
        is_valid(path2models, 'MODELS', 'isdir')
        is_valid(path2dataset, 'DATASET', 'isdir')
        is_valid(path2features, 'FEATURES', 'isdir')
        is_valid(path2checkpoints, 'CHECKPOINTS', 'isdir')

        path2images = path.join(path2dataset, 'images')
        path2task_labels_config = path.join(path2dataset, task_labels_config)
        
        is_valid(path2images, 'images', 'isdir')
        is_valid(path2task_labels_config, 'task_desccription', 'isfile')
        
        task_labels_config = read_description(path2task_labels_config)
        
        ctx.obj['params'] = {
            'backbone': backbone,
            'path2images' : path2images, 
            'img_extension': img_extension, 
            'task_labels_config': task_labels_config, 
            'embeddings_filename': embeddings_filename, 
        } 
        ctx.obj['path2models'] = path2models
        ctx.obj['path2dataset'] = path2dataset
        ctx.obj['path2features'] = path2features 
        ctx.obj['path2checkpoints'] = path2checkpoints

    # end ...!

@router_command.command()
@click.option('--nb_workers', default=4, type=int)
@click.option('--force/--no-force', default=False)
@click.pass_context
def processing(ctx, nb_workers, force):
    processes = []
    try:
        path2models = ctx.obj['path2models']
        path2features = ctx.obj['path2features']

        path2vectorizer = path.join(path2models, f"{ctx.obj['params']['backbone']}.th")
        path2embeddings = path.join(path2features, ctx.obj['params']['embeddings_filename'])

        if path.isfile(path2embeddings):
            logger.debug('embeddings was detected')
            if not force:
                logger.debug('processing stage will be sikpped')
                return 0 

        map_image2labels = ctx.obj['params']['task_labels_config']['map_image2labels']
        path2images = ctx.obj['params']['path2images']
        image_paths = pull_files(path2images, ctx.obj['params']['img_extension'])
        nb_images = len(image_paths)
        logger.debug(f'{nb_images:04d} was found in {path2images}')
        
        indices = np.arange(nb_images)
        intervals = np.array_split(indices, nb_workers)

        queue_ = mp.Queue()
        barrier = mp.Barrier(nb_workers)
        readyness = mp.Event()
        
        map_workerid2nb_images = {}
        for worker_id, sub_indices in enumerate(intervals):
            map_workerid2nb_images[worker_id] = len(sub_indices)
            sub_image_paths = list(op.itemgetter(*sub_indices)(image_paths))
            worker = mp.Process(
                target=vectorize_images, 
                args=(map_image2labels, sub_image_paths, path2vectorizer, worker_id, barrier, readyness, queue_)
            )
            processes.append(worker)
            processes[-1].start()
        # end for ...! 

        readyness.set()  # notifed workers to start processing images 

        features_accumulator = []
        with Progress() as progressor:
            exited_workers = 0 
            map_workerid2taskid = {}
            while exited_workers < nb_workers:
                try:
                    message = queue_.get_nowait()
                    if message['event'] == 'join':
                        task_id = progressor.add_task(
                            description=f'worker : {message["worker_id"]:03d}', 
                            total=map_workerid2nb_images[message['worker_id']]
                        )
                        map_workerid2taskid[message['worker_id']] = task_id 
                    if message['event'] == 'step':
                        progressor.update(
                            task_id=map_workerid2taskid[message['worker_id']], 
                            advance=1
                        )
                        features_accumulator.append((message['worker_id'], message['content']))
                    if message['event'] == 'stop':
                        exited_workers = exited_workers + 1 
                except Exception:  # ignore empty exception 
                    pass
            # end loop monitoring 
        # end context manager processor 
        logger.debug('all workers has done their jobs')
        features_accumulator = sorted(features_accumulator, key=op.itemgetter(0))  # sort by worker_id 
        array_of_embeddings = list(map(op.itemgetter(1), features_accumulator))
        nb_elements = len(array_of_embeddings)

        logger.debug('all workers results were aggregated by the main process')
        serialize(array_of_embeddings, path2embeddings, pickle)
        logger.success(f'{nb_elements:03d} extracted fingerprints were saved')
    except KeyboardInterrupt:
        pass 
    except Exception as e:
        logger.error(e)
    finally:
        logger.success('processing stage was done')
        for prs in processes:
            prs.join()

@router_command.command()
@click.option('--nb_epochs', help='number of epochs', type=int, default=64)
@click.option('--batch_size', help='size of the batch', type=int, default=16)
@click.option('--learning_rate', type=float, help='learning rate', default=1e-3)
@click.option('--optimizer_name', help='name of the optimizer', type=click.Choice(['Adam', 'SGD', 'RMSProp', 'Adagrad', 'Adadelta']), default='Adam')
@click.option('--checkpoint_period', help='period of checkpoint(snapshot)', type=int, default=8)
@click.option('--force/--no-force', default=False)
@click.pass_context
def training(ctx, nb_epochs, batch_size, learning_rate, optimizer_name, checkpoint_period, force):
    try:
        device = 'cpu'
        path2features = ctx.obj['path2features']
        path2checkpoints = ctx.obj['path2checkpoints']
        path2embeddings = path.join(path2features, ctx.obj['params']['embeddings_filename'])

        nb_labels = ctx.obj['params']['task_labels_config']['nb_labels']

        data = deserialize(path2embeddings, pickle)    
        
        X, Y = list(zip(*data))
        X = np.vstack(X)
        Y = np.asarray(Y)

        dataset = TensorDataset(
            th.as_tensor(X).float(),
            th.as_tensor(Y).long()
        )
        data_loader = DataLoader(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True
        )
        
        path2weights = path.join(path2checkpoints, 'snapshot_###.th')
        if path.isfile(path2weights):
            logger.debug(f'a snapshot was found at {path2weights}')
            if not force:
                logger.debug('training stage will be skipped')
                return 0 

        input_layer = MAP_BACKBONE2NB_FEATURES[ctx.obj['params']['backbone']]
        net = MLPModel(layers_config=[input_layer, 256, 128, nb_labels], activations=[1, 1, 0], apply_norm=True)
        net.to(device)
        net.train()

        print(net)

        optimizer = op.attrgetter(optimizer_name)(th.optim)(
            params=net.parameters(), 
            lr=learning_rate
        )

        criterion = nn.CrossEntropyLoss()

        for epoch in range(nb_epochs):
            batch_idx = 0
            for X_batch, Y_batch in track(data_loader, f'epoch : {epoch:03d}'):
                try:
                    logits = net(X_batch.to(device))
                    optimizer.zero_grad()
                    error = criterion(logits, Y_batch.to(device))
                    error.backward()
                    optimizer.step()
                except Exception as e:
                    exception_value = f'local exception : {e}'
                    logger.error(e)
                batch_idx += 1
            # end loop over batchs
            logger.debug(f'[{epoch:03d}/{nb_epochs:03d}] | Loss >> {error.cpu().item():07.3f}')
            if epoch % checkpoint_period == 0:
                path2snapshot = path.join(path2checkpoints, f'snapshot_{epoch:03d}.th')
                th.save(net.cpu(), path2snapshot) 
                logger.success(f'a snapshot was saved at {path2snapshot}')       
        # end loop epochs 
        th.save(net.cpu(), path2weights)
        logger.success('the model was saved...!')
        # end if ... self.path2features is definied ...! => so training can be launched
    except Exception as e:
        exception_value = f'exception : {e}'
        logger.error(exception_value)
    finally:
        logger.debug(f'training stage was done')
# end def traing 


@router_command.command()
@click.option('--server_port', type=int, default=8000)
@click.option('--hostname', default='0.0.0.0')
@click.option('--prefix', help='mounting prefix path', type=str, default='/backend')
@click.pass_context
def serving(ctx, server_port, hostname, prefix):
    ZMQ_INIT = 0
    try:
        device = 'cpu' # move it to command group 

        path2models = ctx.obj['path2models']
        path2checkpoints = ctx.obj['path2checkpoints']

        path2vectorizer = path.join(path2models, f"{ctx.obj['params']['backbone']}.th")
        vectorizer = load_vectorizer(path2vectorizer, device)
        
        index2label = ctx.obj['params']['task_labels_config']['index2label']

        path2predictor = path.join(path2checkpoints, 'snapshot_###.th')
        predictor = th.load(path2predictor)
        predictor.eval()

        zmq_ctx = zmq.Context()
        router_socket = zmq_ctx.socket(zmq.ROUTER)
        router_socket.setsockopt(zmq.LINGER, 0)
        router_socket.bind(f'tcp://*:{ZMQ_SERVER_PORT}')

        router_poller = zmq.Poller()
        router_poller.register(router_socket, zmq.POLLIN)
        ZMQ_INIT = 1

        logger.success('zmq service was initialized')
        app.mount(prefix, ctl)
        api_process = mp.Process(
            target=uvicorn.run, 
            kwargs={'app': app, 'port': server_port, 'host': hostname}
        )

        api_process.start()

        logger.debug(f'server is up and listens at port {ZMQ_SERVER_PORT}')      
        keep_inference = True 
        while keep_inference:
            client_id, _, pickled_image = router_socket.recv_multipart()  # block until there is a message
            try:
                logger.debug('inference got new request for prediction')
                bgr_image = pickle.loads(pickled_image)
                tensor_image = cv2th(bgr_image)
                tensor_image = prepare_image(tensor_image)
                single_batch = tensor_image[None, ...]
                with th.no_grad():
                    embedding = th.flatten(vectorizer(single_batch.to(device)), start_dim=0)
                    logits = th.flatten(predictor(embedding[None, ...]).cpu())
                candidate = th.argmax(logits).item()
                predicted_class = index2label[candidate]
                response = json.dumps({
                    'status_code': 200, 
                    'content': {
                        'message': 'prediction was done successfully',
                        'value': {
                            'distribution': list(zip(index2label, logits.tolist())),
                            'predicted_class': predicted_class
                        }
                    } 
                }).encode()
            except Exception as e:
                exception_value = f'exception during prediction : {e}'
                logger.error(exception_value)
                response = json.dumps({
                    'status_code': 400, 
                    'content': {
                        'message': 'prediction failed',
                        'velue': exception_value
                    } 
                }).encode()
            router_socket.send_multipart([client_id, b'', response])  # send it to remote server 
        # end loop inference ...!
 
    except KeyboardInterrupt as e:
        pass 
    except Exception as e:
        logger.error(e)
    finally:
        if ZMQ_INIT:
            router_poller.unregister(router_socket)
            router_socket.close()
            zmq_ctx.term()
            logger.success('zmq services has removed all ressources')

if __name__ == '__main__':
    try:
        router_command(obj={}) 
    except Exception as e:
        logger.error(e)
