#!/bin/bash
nsys profile -t cuda,mpi,nvtx,cudnn python pretrain_gpt.py
