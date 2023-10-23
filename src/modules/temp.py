# DLAN
in31_branch_1 = Conv(512, 256, 1, 1)(concat_3_5_out)
in32_branch_1 = Conv(512, 256, 1, 1)(concat_3_5_out)

in33_branch_1 = Conv(256, 128, 3, 1)(in33_branch_1)
in33_branch_1_ = Conv(256, 128, 3, 1)(in33_branch_1)

in33_branch_1 = Conv(128, 128, 3, 1)(in33_branch_1_)
in33_branch_1_ = Conv(128, 128, 3, 1)(in33_branch_1)

in35_branch_1 = Conv(128, 128, 3, 1)(in33_branch_1_)
in35_branch_1_ = Conv(128, 128, 3, 1)(in35_branch_1)

in36_branch_1 = Conv(128, 128, 3, 1)(in35_branch_1_)
concat_elan3_branch_1 = torch.cat(
    (in31_branch_1, in32_branch_1, in33_branch_1, in33_branch_1, in35_branch_1, in36_branch_1), 1)
out_elan3_branch_1 = Conv(1024, 256, 1, 1)(concat_elan3_branch_1)

in31_branch_2 = Conv(512, 256, 1, 1)(concat_3_5_out)
in32_branch_2 = Conv(512, 256, 1, 1)(concat_3_5_out)

in33_branch_2 = Conv(256, 128, 3, 1)(in32_branch_2)
in33_branch_2_ = Conv(256, 128, 3, 1)(in33_branch_2)

in33_branch_2 = Conv(128, 128, 3, 1)(in33_branch_2_)
in33_branch_2_ = Conv(128, 128, 3, 1)(in33_branch_2)

in35_branch_2 = Conv(128, 128, 3, 1)(in33_branch_2_)
in35_branch_2_ = Conv(128, 128, 3, 1)(in35_branch_2)

in36_branch_2 = Conv(128, 128, 3, 1)(in35_branch_2_)
concat_elan3_branch_2 = torch.cat(
    (in31_branch_2, in32_branch_2, in33_branch_2, in33_branch_2, in35_branch_2, in36_branch_2), 1)
out_elan3_branch_2 = Conv(1024, 256, 1, 1)(concat_elan3_branch_2)

out_elan3 = out_elan3_branch_1 + out_elan3_branch_2
