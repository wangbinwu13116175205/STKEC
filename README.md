# STKEC
This is the official code of STKEC, "Knowledge Expansion and Consolidation for Continual Traffic Prediction With Expanding Graphs", which is a contintual traffic prediction framework and published in the journal TITS2023.
This code is based on TrafficStream.
# Data
Please download the data from TrafficStream(https://github.com/AprLie/TrafficStream) and follow the same processing method.

# Requirements
conda env create -f STKEC.yaml


# Runing
python main.py --conf STKEC.json --gpuid 4

# TODO

#TODO: Aggregate the day vector of each node, and complete long_pattern=np.load('long_term_path') in line 76 in main.py.

#ToDO: select some nodes with stable patterns to evaluate the effectiveness of knowledge consolidation and select new nodes to to evaluate the effectiveness of knowledge expand.

# Cition
Please cite our paper if you find it useful:

@article{wang2023knowledge,
  title={Knowledge Expansion and Consolidation for Continual Traffic Prediction With Expanding Graphs},
  author={Wang, Binwu and Zhang, Yudong and Shi, Jiahao and Wang, Pengkun and Wang, Xu and Bai, Lei and Wang, Yang},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2023},
  publisher={IEEE}
}


