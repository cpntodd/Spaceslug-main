from pathlib import Path
p=Path('python/spaceslug/backend.py')
s=p.read_text()
start=s.index('                if hasattr(self._library, "spaceslug_tiny_forward_train_embeddings_sgd"):')
end=s.index('                 if hasattr(self._library, "spaceslug_tiny_forward_graph_embedding_training_status"):', start)
block='''                if hasattr(self._library, "spaceslug_tiny_forward_train_embeddings_sgd"):
                    train_embeddings_sgd = self._library.spaceslug_tiny_forward_train_embeddings_sgd
                    train_embeddings_sgd.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_float]
                    train_embeddings_sgd.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_train_positions_sgd"):
                    train_positions_sgd = self._library.spaceslug_tiny_forward_train_positions_sgd
                    train_positions_sgd.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_float]
                    train_positions_sgd.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_readback_positions"):
                    readback_positions = self._library.spaceslug_tiny_forward_readback_positions
                    readback_positions.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
                    readback_positions.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_graph_embedding_training_capability"):
                    graph_embedding_capability = self._library.spaceslug_tiny_forward_graph_embedding_training_capability
                    graph_embedding_capability.argtypes = []
                    graph_embedding_capability.restype = ctypes.c_char_p
'''
p.write_text(s[:start]+block+s[end:])
PY
python3 python/.tmp_fix.py
rm python/.tmp_fix.py