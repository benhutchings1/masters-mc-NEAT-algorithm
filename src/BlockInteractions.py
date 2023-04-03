from mcpi.minecraft import Minecraft as mc
import time
import csv
import numpy as np

class BlockReader():
    def __init__(self, block_path=None):
        self.client = mc.create()
        # Read in block list
        self.blocklist = None
        self.blockmap = None
        if not block_path is None:
            self.blocklist, self.blockmap = self.__read_in_blocks(block_path)
        

    def read_block(self, x, y, z, verbose=False):
        if verbose:
            block = self.client.getBlockWithData(x, y, z)
            return (block.id, block.data)
        else:
            return self.client.getBlock(x, y, z)


    def read_blocks(self, start_coords, end_coords):
            # Read a cube of blocks
            assert type(start_coords) == list or type(end_coords) == list
            assert len(start_coords) == 3 and len(end_coords) == 3
            
            return self.client.getBlocks(
            start_coords[0],
            start_coords[1],
            start_coords[2],
            end_coords[0],
            end_coords[1],
            end_coords[2]
            )
    
    
    def read_blocks_np(self, start_coords, end_coords):
        # Read a cube of blocks
        assert type(start_coords) == list or type(end_coords) == list
        assert len(start_coords) == 3 and len(end_coords) == 3

        # Sort coordinates 
        start_coords, end_coords = self.__sort_coordinates(start_coords, end_coords)

        # Get coordinate vector
        diff = [abs(end_coords[i]-start_coords[i])+1 for i in range(len(start_coords))]
        blocks = np.zeros(shape=(diff[1], diff[0], diff[2])).astype(int)

        # Iterate over y element
        for yi, y in enumerate(range(start_coords[1], start_coords[1] + diff[1])):
            # Iterate over x element
            for xi, x in enumerate(range(start_coords[0], start_coords[0] + diff[0])):
                # Iterate over z element
                arspace = blocks[yi][xi]
                for zi, z in enumerate(range(start_coords[2], start_coords[2] + diff[2])):
                    block = self.read_block(x, y, z, True)
                    strblock = str(block[0])
                    if not block[1] == 0:
                        strblock += f"^{block[1]}"

                    arspace[zi] = self.blockmap[
                        strblock
                    ]
        return blocks                    


    def place_block(self, x, y, z, blockid, subblock=0):
        return self.client.setBlock(x, y, z, blockid, subblock) 
    

    def place_block(self, x, y, z, strblock):
        if "^" in strblock:
            id, subblock = strblock.split("^")
            id, subblock = int(id), int(subblock)
            self.place_block(x, y, z, id, subblock)
        else:
            self.place_block(x, y, z, blockid=int(strblock))


    def place_blocks(self, start_coords, end_coords, blockid, subblock):
        assert type(start_coords) == list or type(end_coords) == list
        assert len(start_coords) == 3 and len(end_coords) == 3

        return self.client.setBlocks(
            start_coords[0],
            start_coords[1],
            start_coords[2],
            end_coords[0],
            end_coords[1],
            end_coords[2],
            blockid,
            subblock
        )

    def place_blocks_np(self, blocks, x, y, z, strblocks=False):
        for yi, y in enumerate(blocks):
            for xi, x in enumerate(y):
                for zi, z in enumerate(x):
                    self.place_block(x+xi, y+yi, z+zi, strblocks=z)

        

    def __read_in_blocks(self, path):
        out = []
        lookup = {}
        with open(path, "r") as fs:
            reader = csv.reader(fs)
            for ri, row in enumerate(reader):
                out.append(row[1])
                lookup[row[1]] = ri
        return out, lookup
                
