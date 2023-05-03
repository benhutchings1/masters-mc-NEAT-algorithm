from mcpi.minecraft import Minecraft as mc
import csv
import numpy as np

class BlockInterface():
    def __init__(self, block_path=None, connect=True):
        if connect:
            self.client = mc.create()
        else:
            self.client = None
            
        # Read in block list
        self.blocklist = None
        self.blockmap = None
        if not block_path is None:
            self.blocklist, self.blockmap = self.__read_in_blocks(block_path)
        
    def connect(self):
        self.client = mc.create()
    
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
    
    
    def read_blocks_np(self, start_coords, end_coords, listid=True):
        # Read a cube of blocks
        assert type(start_coords) == list or type(end_coords) == list
        assert len(start_coords) == 3 and len(end_coords) == 3

        # Sort coordinates 
        start_coords, end_coords = self.__sort_coordinates(start_coords, end_coords)

        # Get coordinate vector
        diff = [abs(end_coords[i]-start_coords[i])+1 for i in range(len(start_coords))]
        if listid:
            blocks = np.zeros(shape=(diff[1], diff[0], diff[2])).astype(int)
        else:
            blocks = np.zeros(shape=(diff[1], diff[0], diff[2])).astype(str)
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

                    if listid:
                        arspace[zi] = self.blockmap[
                            strblock
                        ]
                    else:
                        arspace[zi] = strblock
        return blocks                    


    def place_block_id(self, x, y, z, blockid, subblock=0):
        return self.client.setBlock(x, y, z, blockid, subblock) 
    

    def place_block_str(self, x, y, z, strblock):
        assert type(strblock) == str
        if "^" in strblock:
            id, subblock = strblock.split("^")
            id, subblock = int(id), int(subblock)
            self.place_block_id(x, y, z, id, subblock)
            return
        else:
            self.place_block_id(x, y, z, blockid=int(strblock))    


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

    def place_blocks_np(self, blocks, width, length, x0=0, y0=-60, z0=0, orientation="N",isblocklist=True):
        assert blocks.shape[1] == (2*length + 2*(width - 2))
        # Choose orientation to place blocks in
        axis = self.__orientation_conv(orientation)
        # Map direction changes
        change_dir = {"N":"E", "E":"S", "S":"W", "W":"N"}
    
        side_lengths = [length-1, width-1]*2
        for y, row in enumerate(blocks):
            curr_pos = {"x":x0, "y":y0+y, "z":z0}
            point = 0
            for l in side_lengths:
                side = row[point:point+l]
                point += l
                
                # Place side
                for i, block in enumerate(side):
                    if isblocklist:
                        self.place_block_id(blockid=block, **curr_pos)
                    else:
                        self.place_block_str(strblock=block, **curr_pos)
                    # Update position
                    curr_pos[axis[0]] += axis[1]
                # Change placement direction
                orientation = change_dir[orientation]
                axis = self.__orientation_conv(orientation)
                
                    
                    
    def __orientation_conv(self, orientation):
        if orientation == "N":
            return ["z",1]
        elif orientation == "E":
            return ["x",1]
        elif orientation == "S":
            return ["z",-1]
        elif orientation == "W":
            return ["x",-1]
        else:
            raise AttributeError()
                
            
            
            
                
                    

    def __read_in_blocks(self, path):
        out = []
        lookup = {}
        with open(path, "r") as fs:
            reader = csv.reader(fs)
            for ri, row in enumerate(reader):
                out.append(row[1])
                lookup[row[1]] = ri
        # Remap some lookup values
        remap = [64, 17, 162]        
        for id in remap:
            mpp = lookup[f"{id}"]
            for i in range(20):
                lookup[f"{id}^{i}"] = mpp
        
        return out, lookup

    def __sort_coordinates(self, a, b):
        return [min(ai, bi) for ai, bi in zip(a, b)], \
            [max(ai, bi) for ai, bi in zip(a, b)]
            
    def save_np(self, arr:np.array, path:str):
        with open(path, "w+") as fs:
            wr = csv.writer(fs)
            wr.writerow(arr.shape)
            for y in arr:
                for z in y:
                    wr.writerow(z)
    
    def read_np(self, path:str) -> np.array:
        with open(path, "r") as fs:
            rd = csv.reader(fs)
            shape = ()
            arr = None
            rowy = 0
            rowx=0
            for i, row in enumerate(rd):
                if not i:
                    shape = tuple([int(s) for s in row])
                    arr = np.zeros(shape=shape).astype(int).astype(str)
                else:
                    arr[rowy][rowx] = row
                    rowx += 1
                    if rowx > (shape[1] - 1):
                        rowx = 0
                        rowy += 1
        return arr
    
    def convert_heightmap(self, heightmap, blockidx=1):
        out = np.zeros((np.max(heightmap)+1, heightmap.shape[0], heightmap.shape[1])).astype(int)
        for yi,y in enumerate(heightmap):
            for xi, x in enumerate(y):
                out[x][yi][xi] = blockidx
        return out
        
        