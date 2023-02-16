#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :positions.py
@Data        :2023/01/29 22:32:18
@Version     :1.0
@Author      :Jackokie
@Contact     :jackokie@gmail.com
'''


import numpy as np
import torch


def gen_bs_pos(M, squareLength, rand_pos=True, min_inter_dis=0, min_edge_dis=0):
    """gen_bs_pos _summary_

    _extended_summary_

    Args:
        M (_type_): _description_
        squareLength (_type_): _description_
        rand_pos (bool, optional): _description_. Defaults to True.
        min_inter_dis (int, optional): _description_. Defaults to 0.
        min_edge_dis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    bs_pos = 1e10*torch.rand(M, dtype=torch.complex128)

    if rand_pos==False:

       # Number of APs per dimension on the grid for cell-free and small-cell
        # setups
        nbrAPsPerDim = int(np.sqrt(M))

        # Distance between APs in vertical/horizontal direction
        bs_dis = squareLength / nbrAPsPerDim

        # Deploy APs on the grid
        x = torch.from_numpy(np.arange(bs_dis / 2, squareLength, bs_dis))
        x = torch.expand_copy(x, [nbrAPsPerDim, nbrAPsPerDim])
        y = x.T

        bs_pos = x + 1j * y
        bs_pos = bs_pos.reshape(M)
    else:
        if (min_inter_dis > 0) | (min_edge_dis > 0):
            for i in range(M):
                while True:
                    pos = (torch.rand(1) + 1j * torch.rand(1)) * squareLength
                    if dis_edge_check(bs_pos, pos, squareLength, min_inter_dis, min_edge_dis):
                        bs_pos[i] = pos
                        break
        else:
            bs_pos = (torch.rand(M, 1) + 1j * torch.rand(M, 1)) * squareLength

    bs_pos = np.array(bs_pos)
    bs_pos = np.column_stack([bs_pos.real, bs_pos.imag])
    return bs_pos


def gen_ues_pos(K, width, rand_ues_pos, bs_pos=None, min_ue_ubs_dis=None):
    """gen_ues_pos _summary_

    _extended_summary_

    Args:
        K (_type_): _description_
        width (_type_): _description_
        rand_ues_pos (_type_): _description_
        bs_pos (_type_, optional): _description_. Defaults to None.
        min_ue_ubs_dis (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    ues_pos = torch.zeros(K, dtype=torch.complex128)

    if min_ue_ubs_dis | min_ue_ubs_dis == 0:

        if rand_ues_pos:
            # Generate a random UE location in the area
            ues_pos = (torch.rand(K, dtype=torch.float64) + 1j * torch.rand(K, dtype=torch.float64)) * width
        else:
            assert('We have not realize this function. We will implement it when required.')

    else:
        if rand_ues_pos:
            for k in range(K):

                while True:
                    # Generate a random UE location in the area
                    pos = (torch.rand(1) + 1j * torch.rand(1)) * width
                    if dis_check(pos, bs_pos, min_ue_ubs_dis):
                        ues_pos[k] = pos
                        break
        else:
            # Each UBS will serve defined count of users.
            assert('We have not realize this function. We will implement it when required.')

    ues_pos = np.array(ues_pos)
    ues_pos = np.column_stack([ues_pos.real, ues_pos.imag])
    return ues_pos


def dis_edge_check(bs_pos, pos, width, min_inter_dis, min_edge_dis):
    """dis_edge_check _summary_

    _extended_summary_

    Args:
        bs_pos (_type_): _description_
        pos (_type_): _description_
        width (_type_): _description_
        min_inter_dis (_type_): _description_
        min_edge_dis (_type_): _description_

    Returns:
        _type_: _description_
    """
    idx = dis_check(pos, bs_pos, min_inter_dis) & edge_check(pos, width, min_edge_dis)
    return torch.sum(idx) == len(bs_pos)


def dis_check(pos, bs_pos, min_inter_dis):
    """dis_check _summary_

    _extended_summary_

    Args:
        pos (_type_): _description_
        bs_pos (_type_): _description_
        min_inter_dis (_type_): _description_

    Returns:
        _type_: _description_
    """
    return torch.abs(pos - bs_pos) < min_inter_dis


def edge_check(pos, width, min_edge_dis):
    """edge_check _summary_

    _extended_summary_

    Args:
        pos (_type_): _description_
        width (_type_): _description_
        min_edge_dis (_type_): _description_

    Returns:
        _type_: _description_
    """
    edge_chk = (torch.abs(pos.real) > min_edge_dis) & (torch.abs(width - pos.real) > min_edge_dis) \
        & (torch.abs(pos.imag) > min_edge_dis) & (torch.abs(width - pos.imag) > min_edge_dis)
    return edge_chk
