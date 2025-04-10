import asyncio
import logging
import traceback
import einops

import cv2
import numpy as np
from typing import Dict, Any

import numpy as np
import torch
import websockets.asyncio.server
import websockets.frames

import gr00t.utils.msgpack_utils as msgpack_utils 
from gr00t.model.policy import Gr00tPolicy

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.
    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: Gr00tPolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}        

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()
    
    async def preprocess_observation(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input
        {
            "state": np.ndarray(shape=(17,)), 
            "images": {
                "front": np.uint8 array (H,W,3),
                "wrist_right": np.uint8 array (H,W,3)
            },
            "prompt": str
        }
        
        Output
        {
            "video.front": uint8 array (1,360,640,3),
            "video.wrist_right": uint8 array (1,360,640,3),
            "state.left_arm": float32 array (1,6),
            "state.right_arm": float32 array (1,6),
            "state.vacuum": float32 array (1,1),
            "state.waist": float32 array (1,4),
            "annotation.human.action.task_description": [str]
        }
        """
        processed = {}

        for cam_name in raw_input["images"].keys():
            img = raw_input["images"][cam_name]
            batch_img = np.expand_dims(img, axis=0)  
            processed[f"video.{cam_name}"] = batch_img

        state = raw_input["state"].astype(np.float32)
        processed.update({
            "state.left_arm": state[:6].reshape(1,6),
            "state.right_arm": state[6:12].reshape(1,6),
            "state.vacuum": state[12:13].reshape(1,1),
            "state.waist": state[13:17].reshape(1,4),
        })
        
        processed["annotation.human.action.task_description"] = [raw_input["prompt"]]
        
        return processed

    async def merge_actions(self, action_dict):
        left_arm = action_dict['action.left_arm']
        right_arm = action_dict['action.right_arm']
        vacuum = action_dict['action.vacuum']
        waist = action_dict['action.waist']

        if vacuum.ndim == 1:
            vacuum = vacuum.reshape(-1, 1)

        merged_actions = np.concatenate(
            [left_arm, right_arm, vacuum, waist],
            axis=1
        )
        
        return merged_actions


    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_utils.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                obs = msgpack_utils.unpackb(await websocket.recv())

                obs = await self.preprocess_observation(obs)
                # print(f"obs: {obs}")

                action = self._policy.get_action(obs)
                # print("inference once with action:",  next(iter(action.values())).shape, action)

                merged_actions = await self.merge_actions(action)
                print("inference once with action: ", merged_actions)

                res = {"actions": merged_actions}
                await websocket.send(packer.pack(res))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise