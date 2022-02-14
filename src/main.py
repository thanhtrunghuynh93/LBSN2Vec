"""Running the Splitter."""

import torch
from param_parser import parameter_parser
from splitter import SplitterTrainer
from utils import tab_printer, graph_reader
import csv
from scipy.io import loadmat

def main():
    """
    Parsing command line parameters.
    Reading data, embedding base graph, creating persona graph and learning a splitter.
    Saving the persona mapping and the embedding.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)
    # graph data fromat?
    graph = graph_reader(args.edge_path)
    graph_friend = graph_reader(args.edge_path_friend)
    mat = loadmat('dataset/cleaned_{}.mat'.format(args.lbsn))

    with open(args.listPOI) as f:
        reader = csv.reader(f)
        # print(reader)
        listPOI = [int(i[0]) for i in reader]
    location_Dict = dict()
    with open(args.location_dict) as f:
        for line in f:
            a,b = line.split("\n")[0].split(("\t"))
            a,b = int(a),int(b)
            location_Dict[a] = b
    # print(listPOI)
    # exit()
    trainer = SplitterTrainer(graph,graph_friend,listPOI,mat,location_Dict, args)
    trainer.fit()
    trainer.save_embedding()
    trainer.save_persona_graph_mapping()

if __name__ == "__main__":
    main()
