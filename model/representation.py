import torch
import torch.nn as nn
from operator import itemgetter

class HashTable(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        self.table_size = cfg['table_size']   # e.g. 2**16
        self.hash_table = dict()
        self.primes = [73856093, 19349663, 83492791]

    def _hash(self, coords):
        """Simple spatial hash function for 3D coordinates.
        Coordinates are multiplied by a number and combined using XOR."""
        # Prime numbers for hashing
        h = coords[:, 0] * self.primes[0] + coords[:, 1] * self.primes[1] + coords[:, 2] * self.primes[2]
        print(f'h: {h}')
        
        # Check the hash is within the bounds of the hash table
        return tuple((h % self.table_size).tolist())
    
    # Function to insert data into the hash table
    def insert(self, coords, features):
        # Get the hash for the given 3D point
        h = self._hash(coords)
        print(h)
        # Insert the data into the hash table
        self.hash_table = {key: value for key, value in zip(h, features)}

    # Function to retrieve data from the hash table
    def query(self, coords):
        # Get the hash for the given 3D point
        h = self._hash(coords)
        # Retrieve the data from the hash table
        return [self.hash_table.get(key, None) for key in h]
    
    def display(self):
        # Display the contents of the hash table.
        for key, values in self.hash_table.items():
            print(f"{key}: {values}")


if __name__=='__main__':
    cfg = dict()
    cfg['table_size'] = 2**4
    ht = HashTable(cfg)

    torch.manual_seed(42)
    coords = torch.rand(2, 3)
    features = torch.rand(2, 4)

    ht.insert(coords, features)
    values = ht.query(coords)

    print(values)