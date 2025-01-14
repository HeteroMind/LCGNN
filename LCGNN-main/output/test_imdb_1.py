nohup: ignoring input
2024-12-26 08:14:23,557:INFO::=============== Search Args:
Namespace(dataset='IMDB', feats_type=6, gnn_model='simpleHGN', valid_attributed_type=0, hidden_dim=64, num_heads=8, attn_vec_dim=128, patience=30, batch_size=8, batch_size_test=64, repeat=5, save_postfix='IMDB', feats_opt='0111', cuda=True, l2norm=False, time_line='2024-12-26-08-14-23', edge_feats=64, rnn_type='RotatE0', seed=123, use_adamw=False, neighbor_samples=100, att_comp_dim=64, use_norm=False, schedule_step_retrain=500, patience_search=30, patience_retrain=30, is_use_type_linear='False', is_use_SGD='False', is_use_dropout='False', momentum=0.9, inner_epoch=1, use_minibatch=False, useSGD=False, useTypeLinear=False, usedropout=False, usebn=False, use_bn=False, use_5seeds=True, cur_repeat=False, no_use_fixseeds=False, lr_rate_min=3e-05, beta_1=1.0, lr=0.0005, num_layers=2, complete_num_layers=2, dropout=0.5, weight_decay=0.0001, slope=0.1, grad_clip=5, max_num_views=4, complete_epochs=1, search_epoch=350, retrain_epoch=500, intralayers=2, last_hidden_dim=512, logger=<Logger log_output (INFO)>)
2024-12-26 08:14:31,148:INFO::node_type_num: 4
/home/yyj/miniconda3/envs/yyj_mdnnac/lib/python3.9/site-packages/dgl/heterograph.py:72: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning('Recommend creating graphs by `dgl.graph(data)`'
2024-12-26 08:14:46,691:INFO::=============== Prepare basic data stage finish, use 23.13451337814331 time.
2024-12-26 08:14:50,550:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:14:54,333:INFO::Epoch_batch_00000 | lr 0.0005 |Train_Loss 0.7066 |  total_loss_consistency1 0.1813 | Loss 0.8879 | Val_Loss 0.6405| Train Time(s) 3.5625| Val Time(s) 0.0713 | Time(s) 3.6338
2024-12-26 08:14:55,888:INFO::Epoch_batch_00001 | lr 0.0005 |Train_Loss 0.6431 |  total_loss_consistency1 0.1694 | Loss 0.8126 | Val_Loss 0.6351| Train Time(s) 1.1640| Val Time(s) 0.1447 | Time(s) 1.3086
2024-12-26 08:14:55,889:INFO::Validation loss decreased (inf --> 0.635115).  Saving model ...
2024-12-26 08:14:57,205:INFO::Epoch_batch_00002 | lr 0.0005 |Train_Loss 0.6319 |  total_loss_consistency1 0.1601 | Loss 0.7921 | Val_Loss 0.6329| Train Time(s) 0.9570| Val Time(s) 0.1129 | Time(s) 1.0699
2024-12-26 08:14:57,206:INFO::Validation loss decreased (0.635115 --> 0.632908).  Saving model ...
2024-12-26 08:14:58,500:INFO::Epoch_batch_00003 | lr 0.0005 |Train_Loss 0.6320 |  total_loss_consistency1 0.1531 | Loss 0.7852 | Val_Loss 0.6308| Train Time(s) 0.9344| Val Time(s) 0.1129 | Time(s) 1.0473
2024-12-26 08:14:58,501:INFO::Validation loss decreased (0.632908 --> 0.630777).  Saving model ...
2024-12-26 08:15:00,028:INFO::Epoch_batch_00004 | lr 0.0005 |Train_Loss 0.6301 |  total_loss_consistency1 0.1479 | Loss 0.7780 | Val_Loss 0.6287| Train Time(s) 1.1290| Val Time(s) 0.1496 | Time(s) 1.2787
2024-12-26 08:15:00,029:INFO::Validation loss decreased (0.630777 --> 0.628739).  Saving model ...
2024-12-26 08:15:01,337:INFO::Epoch_batch_00005 | lr 0.0005 |Train_Loss 0.6279 |  total_loss_consistency1 0.1429 | Loss 0.7709 | Val_Loss 0.6269| Train Time(s) 0.9719| Val Time(s) 0.0841 | Time(s) 1.0559
2024-12-26 08:15:01,337:INFO::Validation loss decreased (0.628739 --> 0.626880).  Saving model ...
2024-12-26 08:15:02,632:INFO::Epoch_batch_00006 | lr 0.0005 |Train_Loss 0.6259 |  total_loss_consistency1 0.1383 | Loss 0.7642 | Val_Loss 0.6252| Train Time(s) 0.9418| Val Time(s) 0.1088 | Time(s) 1.0506
2024-12-26 08:15:02,633:INFO::Validation loss decreased (0.626880 --> 0.625192).  Saving model ...
2024-12-26 08:15:04,179:INFO::Epoch_batch_00007 | lr 0.0005 |Train_Loss 0.6245 |  total_loss_consistency1 0.1344 | Loss 0.7589 | Val_Loss 0.6236| Train Time(s) 1.1797| Val Time(s) 0.1246 | Time(s) 1.3043
2024-12-26 08:15:04,179:INFO::Validation loss decreased (0.625192 --> 0.623626).  Saving model ...
2024-12-26 08:15:05,480:INFO::Epoch_batch_00008 | lr 0.0005 |Train_Loss 0.6232 |  total_loss_consistency1 0.1312 | Loss 0.7545 | Val_Loss 0.6220| Train Time(s) 0.9460| Val Time(s) 0.1127 | Time(s) 1.0587
2024-12-26 08:15:05,480:INFO::Validation loss decreased (0.623626 --> 0.622020).  Saving model ...
2024-12-26 08:15:06,785:INFO::Epoch_batch_00009 | lr 0.0005 |Train_Loss 0.6206 |  total_loss_consistency1 0.1281 | Loss 0.7487 | Val_Loss 0.6204| Train Time(s) 0.9429| Val Time(s) 0.1133 | Time(s) 1.0562
2024-12-26 08:15:06,785:INFO::Validation loss decreased (0.622020 --> 0.620372).  Saving model ...
2024-12-26 08:15:08,319:INFO::Epoch_batch_00010 | lr 0.0005 |Train_Loss 0.6188 |  total_loss_consistency1 0.1260 | Loss 0.7448 | Val_Loss 0.6187| Train Time(s) 1.1573| Val Time(s) 0.1321 | Time(s) 1.2895
2024-12-26 08:15:08,320:INFO::Validation loss decreased (0.620372 --> 0.618671).  Saving model ...
2024-12-26 08:15:09,614:INFO::Epoch_batch_00011 | lr 0.0005 |Train_Loss 0.6177 |  total_loss_consistency1 0.1240 | Loss 0.7417 | Val_Loss 0.6169| Train Time(s) 0.9420| Val Time(s) 0.1080 | Time(s) 1.0500
2024-12-26 08:15:09,615:INFO::Validation loss decreased (0.618671 --> 0.616914).  Saving model ...
2024-12-26 08:15:10,923:INFO::Epoch_batch_00012 | lr 0.0005 |Train_Loss 0.6149 |  total_loss_consistency1 0.1222 | Loss 0.7370 | Val_Loss 0.6151| Train Time(s) 0.9562| Val Time(s) 0.0827 | Time(s) 1.0389
2024-12-26 08:15:10,923:INFO::Validation loss decreased (0.616914 --> 0.615112).  Saving model ...
2024-12-26 08:15:12,477:INFO::Epoch_batch_00013 | lr 0.0005 |Train_Loss 0.6136 |  total_loss_consistency1 0.1204 | Loss 0.7340 | Val_Loss 0.6132| Train Time(s) 1.1571| Val Time(s) 0.1482 | Time(s) 1.3052
2024-12-26 08:15:12,478:INFO::Validation loss decreased (0.615112 --> 0.613226).  Saving model ...
2024-12-26 08:15:13,774:INFO::Epoch_batch_00014 | lr 0.0005 |Train_Loss 0.6126 |  total_loss_consistency1 0.1187 | Loss 0.7313 | Val_Loss 0.6112| Train Time(s) 0.9547| Val Time(s) 0.1008 | Time(s) 1.0555
2024-12-26 08:15:13,774:INFO::Validation loss decreased (0.613226 --> 0.611174).  Saving model ...
2024-12-26 08:15:15,082:INFO::Epoch_batch_00015 | lr 0.0005 |Train_Loss 0.6077 |  total_loss_consistency1 0.1174 | Loss 0.7251 | Val_Loss 0.6089| Train Time(s) 0.9591| Val Time(s) 0.1058 | Time(s) 1.0649
2024-12-26 08:15:15,082:INFO::Validation loss decreased (0.611174 --> 0.608874).  Saving model ...
2024-12-26 08:15:16,628:INFO::Epoch_batch_00016 | lr 0.0005 |Train_Loss 0.6051 |  total_loss_consistency1 0.1159 | Loss 0.7211 | Val_Loss 0.6065| Train Time(s) 1.1247| Val Time(s) 0.1785 | Time(s) 1.3032
2024-12-26 08:15:16,628:INFO::Validation loss decreased (0.608874 --> 0.606475).  Saving model ...
2024-12-26 08:15:17,928:INFO::Epoch_batch_00017 | lr 0.0005 |Train_Loss 0.6030 |  total_loss_consistency1 0.1147 | Loss 0.7178 | Val_Loss 0.6039| Train Time(s) 0.9405| Val Time(s) 0.1127 | Time(s) 1.0532
2024-12-26 08:15:17,928:INFO::Validation loss decreased (0.606475 --> 0.603947).  Saving model ...
2024-12-26 08:15:19,218:INFO::Epoch_batch_00018 | lr 0.0005 |Train_Loss 0.5998 |  total_loss_consistency1 0.1133 | Loss 0.7132 | Val_Loss 0.6013| Train Time(s) 0.9306| Val Time(s) 0.1120 | Time(s) 1.0426
2024-12-26 08:15:19,218:INFO::Validation loss decreased (0.603947 --> 0.601297).  Saving model ...
2024-12-26 08:15:20,689:INFO::Epoch_batch_00019 | lr 0.0005 |Train_Loss 0.5963 |  total_loss_consistency1 0.1124 | Loss 0.7087 | Val_Loss 0.5987| Train Time(s) 1.0509| Val Time(s) 0.1771 | Time(s) 1.2280
2024-12-26 08:15:20,689:INFO::Validation loss decreased (0.601297 --> 0.598664).  Saving model ...
2024-12-26 08:15:22,006:INFO::Epoch_batch_00020 | lr 0.0005 |Train_Loss 0.5924 |  total_loss_consistency1 0.1118 | Loss 0.7042 | Val_Loss 0.5960| Train Time(s) 0.9469| Val Time(s) 0.1104 | Time(s) 1.0573
2024-12-26 08:15:22,006:INFO::Validation loss decreased (0.598664 --> 0.596037).  Saving model ...
2024-12-26 08:15:23,312:INFO::Epoch_batch_00021 | lr 0.0005 |Train_Loss 0.5888 |  total_loss_consistency1 0.1109 | Loss 0.6997 | Val_Loss 0.5935| Train Time(s) 0.9755| Val Time(s) 0.0771 | Time(s) 1.0527
2024-12-26 08:15:23,312:INFO::Validation loss decreased (0.596037 --> 0.593509).  Saving model ...
2024-12-26 08:15:24,845:INFO::Epoch_batch_00022 | lr 0.0005 |Train_Loss 0.5849 |  total_loss_consistency1 0.1103 | Loss 0.6952 | Val_Loss 0.5911| Train Time(s) 1.0924| Val Time(s) 0.1843 | Time(s) 1.2768
2024-12-26 08:15:24,846:INFO::Validation loss decreased (0.593509 --> 0.591069).  Saving model ...
2024-12-26 08:15:26,099:INFO::Epoch_batch_00023 | lr 0.0005 |Train_Loss 0.5823 |  total_loss_consistency1 0.1095 | Loss 0.6917 | Val_Loss 0.5888| Train Time(s) 0.9503| Val Time(s) 0.1039 | Time(s) 1.0542
2024-12-26 08:15:26,100:INFO::Validation loss decreased (0.591069 --> 0.588838).  Saving model ...
2024-12-26 08:15:27,410:INFO::Epoch_batch_00024 | lr 0.0005 |Train_Loss 0.5814 |  total_loss_consistency1 0.1087 | Loss 0.6900 | Val_Loss 0.5874| Train Time(s) 0.9427| Val Time(s) 0.1131 | Time(s) 1.0558
2024-12-26 08:15:27,410:INFO::Validation loss decreased (0.588838 --> 0.587396).  Saving model ...
2024-12-26 08:15:28,955:INFO::Epoch_batch_00025 | lr 0.0005 |Train_Loss 0.5770 |  total_loss_consistency1 0.1078 | Loss 0.6848 | Val_Loss 0.5866| Train Time(s) 1.1333| Val Time(s) 0.1678 | Time(s) 1.3011
2024-12-26 08:15:28,956:INFO::Validation loss decreased (0.587396 --> 0.586570).  Saving model ...
2024-12-26 08:15:30,260:INFO::Epoch_batch_00026 | lr 0.0005 |Train_Loss 0.5726 |  total_loss_consistency1 0.1072 | Loss 0.6799 | Val_Loss 0.5862| Train Time(s) 0.9446| Val Time(s) 0.1124 | Time(s) 1.0570
2024-12-26 08:15:30,261:INFO::Validation loss decreased (0.586570 --> 0.586198).  Saving model ...
2024-12-26 08:15:31,585:INFO::Epoch_batch_00027 | lr 0.0005 |Train_Loss 0.5715 |  total_loss_consistency1 0.1063 | Loss 0.6778 | Val_Loss 0.5858| Train Time(s) 0.9642| Val Time(s) 0.1120 | Time(s) 1.0761
2024-12-26 08:15:31,585:INFO::Validation loss decreased (0.586198 --> 0.585796).  Saving model ...
2024-12-26 08:15:33,131:INFO::Epoch_batch_00028 | lr 0.0005 |Train_Loss 0.5680 |  total_loss_consistency1 0.1056 | Loss 0.6737 | Val_Loss 0.5854| Train Time(s) 1.1583| Val Time(s) 0.1384 | Time(s) 1.2967
2024-12-26 08:15:33,131:INFO::Validation loss decreased (0.585796 --> 0.585395).  Saving model ...
2024-12-26 08:15:34,433:INFO::Epoch_batch_00029 | lr 0.0005 |Train_Loss 0.5658 |  total_loss_consistency1 0.1047 | Loss 0.6705 | Val_Loss 0.5846| Train Time(s) 0.9608| Val Time(s) 0.0813 | Time(s) 1.0420
2024-12-26 08:15:34,433:INFO::Validation loss decreased (0.585395 --> 0.584606).  Saving model ...
2024-12-26 08:15:35,728:INFO::Epoch_batch_00030 | lr 0.0005 |Train_Loss 0.5667 |  total_loss_consistency1 0.1038 | Loss 0.6705 | Val_Loss 0.5840| Train Time(s) 0.9711| Val Time(s) 0.0762 | Time(s) 1.0473
2024-12-26 08:15:35,728:INFO::Validation loss decreased (0.584606 --> 0.584025).  Saving model ...
2024-12-26 08:15:37,272:INFO::Epoch_batch_00031 | lr 0.0005 |Train_Loss 0.5613 |  total_loss_consistency1 0.1031 | Loss 0.6644 | Val_Loss 0.5836| Train Time(s) 1.1796| Val Time(s) 0.1202 | Time(s) 1.2998
2024-12-26 08:15:37,272:INFO::Validation loss decreased (0.584025 --> 0.583622).  Saving model ...
2024-12-26 08:15:38,568:INFO::Epoch_batch_00032 | lr 0.0005 |Train_Loss 0.5609 |  total_loss_consistency1 0.1022 | Loss 0.6631 | Val_Loss 0.5829| Train Time(s) 0.9521| Val Time(s) 0.1043 | Time(s) 1.0563
2024-12-26 08:15:38,568:INFO::Validation loss decreased (0.583622 --> 0.582931).  Saving model ...
2024-12-26 08:15:39,869:INFO::Epoch_batch_00033 | lr 0.0005 |Train_Loss 0.5601 |  total_loss_consistency1 0.1013 | Loss 0.6614 | Val_Loss 0.5823| Train Time(s) 0.9508| Val Time(s) 0.1039 | Time(s) 1.0547
2024-12-26 08:15:39,869:INFO::Validation loss decreased (0.582931 --> 0.582335).  Saving model ...
2024-12-26 08:15:41,426:INFO::Epoch_batch_00034 | lr 0.0005 |Train_Loss 0.5569 |  total_loss_consistency1 0.1005 | Loss 0.6575 | Val_Loss 0.5821| Train Time(s) 1.1967| Val Time(s) 0.1157 | Time(s) 1.3124
2024-12-26 08:15:41,426:INFO::Validation loss decreased (0.582335 --> 0.582060).  Saving model ...
2024-12-26 08:15:42,719:INFO::Epoch_batch_00035 | lr 0.0005 |Train_Loss 0.5543 |  total_loss_consistency1 0.0997 | Loss 0.6540 | Val_Loss 0.5818| Train Time(s) 0.9391| Val Time(s) 0.1122 | Time(s) 1.0513
2024-12-26 08:15:42,719:INFO::Validation loss decreased (0.582060 --> 0.581778).  Saving model ...
2024-12-26 08:15:44,024:INFO::Epoch_batch_00036 | lr 0.0005 |Train_Loss 0.5541 |  total_loss_consistency1 0.0991 | Loss 0.6531 | Val_Loss 0.5815| Train Time(s) 0.9592| Val Time(s) 0.1082 | Time(s) 1.0674
2024-12-26 08:15:44,024:INFO::Validation loss decreased (0.581778 --> 0.581455).  Saving model ...
2024-12-26 08:15:45,537:INFO::Epoch_batch_00037 | lr 0.0005 |Train_Loss 0.5512 |  total_loss_consistency1 0.0983 | Loss 0.6495 | Val_Loss 0.5807| Train Time(s) 1.1947| Val Time(s) 0.0824 | Time(s) 1.2771
2024-12-26 08:15:45,538:INFO::Validation loss decreased (0.581455 --> 0.580651).  Saving model ...
2024-12-26 08:15:46,836:INFO::Epoch_batch_00038 | lr 0.0005 |Train_Loss 0.5521 |  total_loss_consistency1 0.0974 | Loss 0.6496 | Val_Loss 0.5801| Train Time(s) 0.9841| Val Time(s) 0.0708 | Time(s) 1.0549
2024-12-26 08:15:46,836:INFO::Validation loss decreased (0.580651 --> 0.580086).  Saving model ...
2024-12-26 08:15:48,128:INFO::Epoch_batch_00039 | lr 0.0005 |Train_Loss 0.5498 |  total_loss_consistency1 0.0966 | Loss 0.6464 | Val_Loss 0.5798| Train Time(s) 0.9667| Val Time(s) 0.0888 | Time(s) 1.0555
2024-12-26 08:15:48,129:INFO::Validation loss decreased (0.580086 --> 0.579812).  Saving model ...
2024-12-26 08:15:49,665:INFO::Epoch_batch_00040 | lr 0.0005 |Train_Loss 0.5477 |  total_loss_consistency1 0.0959 | Loss 0.6436 | Val_Loss 0.5796| Train Time(s) 1.1858| Val Time(s) 0.1142 | Time(s) 1.3000
2024-12-26 08:15:49,666:INFO::Validation loss decreased (0.579812 --> 0.579633).  Saving model ...
2024-12-26 08:15:50,970:INFO::Epoch_batch_00041 | lr 0.0005 |Train_Loss 0.5467 |  total_loss_consistency1 0.0950 | Loss 0.6416 | Val_Loss 0.5795| Train Time(s) 0.9501| Val Time(s) 0.1033 | Time(s) 1.0534
2024-12-26 08:15:50,971:INFO::Validation loss decreased (0.579633 --> 0.579515).  Saving model ...
2024-12-26 08:15:52,273:INFO::Epoch_batch_00042 | lr 0.0005 |Train_Loss 0.5444 |  total_loss_consistency1 0.0941 | Loss 0.6385 | Val_Loss 0.5794| Train Time(s) 0.9551| Val Time(s) 0.1016 | Time(s) 1.0567
2024-12-26 08:15:52,273:INFO::Validation loss decreased (0.579515 --> 0.579448).  Saving model ...
2024-12-26 08:15:53,824:INFO::Epoch_batch_00043 | lr 0.0005 |Train_Loss 0.5442 |  total_loss_consistency1 0.0934 | Loss 0.6376 | Val_Loss 0.5793| Train Time(s) 1.1822| Val Time(s) 0.1306 | Time(s) 1.3129
2024-12-26 08:15:53,824:INFO::Validation loss decreased (0.579448 --> 0.579274).  Saving model ...
2024-12-26 08:15:54,877:INFO::Epoch_batch_00044 | lr 0.0005 |Train_Loss 0.5436 |  total_loss_consistency1 0.0926 | Loss 0.6362 | Val_Loss 0.5793| Train Time(s) 0.9375| Val Time(s) 0.1120 | Time(s) 1.0495
2024-12-26 08:15:54,877:INFO::EarlyStopping counter: 1 out of 30
2024-12-26 08:15:55,918:INFO::Epoch_batch_00045 | lr 0.0005 |Train_Loss 0.5432 |  total_loss_consistency1 0.0917 | Loss 0.6349 | Val_Loss 0.5794| Train Time(s) 0.9258| Val Time(s) 0.1127 | Time(s) 1.0385
2024-12-26 08:15:55,918:INFO::EarlyStopping counter: 2 out of 30
2024-12-26 08:15:57,199:INFO::Epoch_batch_00046 | lr 0.0005 |Train_Loss 0.5411 |  total_loss_consistency1 0.0909 | Loss 0.6320 | Val_Loss 0.5792| Train Time(s) 0.8940| Val Time(s) 0.1503 | Time(s) 1.0443
2024-12-26 08:15:57,200:INFO::Validation loss decreased (0.579274 --> 0.579217).  Saving model ...
2024-12-26 08:15:58,551:INFO::Epoch_batch_00047 | lr 0.0005 |Train_Loss 0.5395 |  total_loss_consistency1 0.0900 | Loss 0.6295 | Val_Loss 0.5790| Train Time(s) 1.0076| Val Time(s) 0.1072 | Time(s) 1.1148
2024-12-26 08:15:58,552:INFO::Validation loss decreased (0.579217 --> 0.579045).  Saving model ...
2024-12-26 08:15:59,854:INFO::Epoch_batch_00048 | lr 0.0005 |Train_Loss 0.5398 |  total_loss_consistency1 0.0893 | Loss 0.6291 | Val_Loss 0.5788| Train Time(s) 0.9445| Val Time(s) 0.1105 | Time(s) 1.0550
2024-12-26 08:15:59,855:INFO::Validation loss decreased (0.579045 --> 0.578757).  Saving model ...
2024-12-26 08:16:01,116:INFO::Epoch_batch_00049 | lr 0.0005 |Train_Loss 0.5384 |  total_loss_consistency1 0.0885 | Loss 0.6269 | Val_Loss 0.5786| Train Time(s) 0.9181| Val Time(s) 0.1031 | Time(s) 1.0213
2024-12-26 08:16:01,116:INFO::Validation loss decreased (0.578757 --> 0.578617).  Saving model ...
2024-12-26 08:16:02,545:INFO::Epoch_batch_00050 | lr 0.0005 |Train_Loss 0.5366 |  total_loss_consistency1 0.0878 | Loss 0.6243 | Val_Loss 0.5786| Train Time(s) 1.1151| Val Time(s) 0.0764 | Time(s) 1.1914
2024-12-26 08:16:02,545:INFO::Validation loss decreased (0.578617 --> 0.578594).  Saving model ...
2024-12-26 08:16:03,842:INFO::Epoch_batch_00051 | lr 0.0005 |Train_Loss 0.5370 |  total_loss_consistency1 0.0868 | Loss 0.6238 | Val_Loss 0.5784| Train Time(s) 0.9758| Val Time(s) 0.0812 | Time(s) 1.0570
2024-12-26 08:16:03,842:INFO::Validation loss decreased (0.578594 --> 0.578437).  Saving model ...
2024-12-26 08:16:05,101:INFO::Epoch_batch_00052 | lr 0.0005 |Train_Loss 0.5351 |  total_loss_consistency1 0.0859 | Loss 0.6210 | Val_Loss 0.5784| Train Time(s) 0.9133| Val Time(s) 0.1087 | Time(s) 1.0220
2024-12-26 08:16:05,101:INFO::Validation loss decreased (0.578437 --> 0.578374).  Saving model ...
2024-12-26 08:16:06,586:INFO::Epoch_batch_00053 | lr 0.0005 |Train_Loss 0.5343 |  total_loss_consistency1 0.0851 | Loss 0.6194 | Val_Loss 0.5783| Train Time(s) 1.1554| Val Time(s) 0.0888 | Time(s) 1.2443
2024-12-26 08:16:06,586:INFO::Validation loss decreased (0.578374 --> 0.578299).  Saving model ...
2024-12-26 08:16:07,912:INFO::Epoch_batch_00054 | lr 0.0005 |Train_Loss 0.5342 |  total_loss_consistency1 0.0841 | Loss 0.6184 | Val_Loss 0.5783| Train Time(s) 0.9810| Val Time(s) 0.1112 | Time(s) 1.0922
2024-12-26 08:16:07,913:INFO::Validation loss decreased (0.578299 --> 0.578297).  Saving model ...
2024-12-26 08:16:08,940:INFO::Epoch_batch_00055 | lr 0.0005 |Train_Loss 0.5335 |  total_loss_consistency1 0.0831 | Loss 0.6167 | Val_Loss 0.5787| Train Time(s) 0.9212| Val Time(s) 0.1037 | Time(s) 1.0248
2024-12-26 08:16:08,941:INFO::EarlyStopping counter: 1 out of 30
2024-12-26 08:16:10,481:INFO::Epoch_batch_00056 | lr 0.0005 |Train_Loss 0.5331 |  total_loss_consistency1 0.0822 | Loss 0.6153 | Val_Loss 0.5782| Train Time(s) 1.2020| Val Time(s) 0.1022 | Time(s) 1.3042
2024-12-26 08:16:10,482:INFO::Validation loss decreased (0.578297 --> 0.578177).  Saving model ...
2024-12-26 08:16:11,538:INFO::Epoch_batch_00057 | lr 0.0005 |Train_Loss 0.5328 |  total_loss_consistency1 0.0809 | Loss 0.6137 | Val_Loss 0.5783| Train Time(s) 0.9413| Val Time(s) 0.1128 | Time(s) 1.0541
2024-12-26 08:16:11,539:INFO::EarlyStopping counter: 1 out of 30
2024-12-26 08:16:12,594:INFO::Epoch_batch_00058 | lr 0.0005 |Train_Loss 0.5314 |  total_loss_consistency1 0.0801 | Loss 0.6116 | Val_Loss 0.5782| Train Time(s) 0.9393| Val Time(s) 0.1131 | Time(s) 1.0524
2024-12-26 08:16:12,594:INFO::EarlyStopping counter: 2 out of 30
2024-12-26 08:16:13,724:INFO::Epoch_batch_00059 | lr 0.0005 |Train_Loss 0.5306 |  total_loss_consistency1 0.0789 | Loss 0.6095 | Val_Loss 0.5786| Train Time(s) 0.9466| Val Time(s) 0.1785 | Time(s) 1.1251
2024-12-26 08:16:13,724:INFO::EarlyStopping counter: 3 out of 30
2024-12-26 08:16:14,930:INFO::Epoch_batch_00060 | lr 0.0005 |Train_Loss 0.5314 |  total_loss_consistency1 0.0778 | Loss 0.6092 | Val_Loss 0.5787| Train Time(s) 1.0895| Val Time(s) 0.1131 | Time(s) 1.2026
2024-12-26 08:16:14,930:INFO::EarlyStopping counter: 4 out of 30
2024-12-26 08:16:16,008:INFO::Epoch_batch_00061 | lr 0.0005 |Train_Loss 0.5298 |  total_loss_consistency1 0.0764 | Loss 0.6062 | Val_Loss 0.5789| Train Time(s) 0.9575| Val Time(s) 0.1176 | Time(s) 1.0751
2024-12-26 08:16:16,008:INFO::EarlyStopping counter: 5 out of 30
2024-12-26 08:16:17,017:INFO::Epoch_batch_00062 | lr 0.0005 |Train_Loss 0.5299 |  total_loss_consistency1 0.0752 | Loss 0.6051 | Val_Loss 0.5792| Train Time(s) 0.9093| Val Time(s) 0.0992 | Time(s) 1.0085
2024-12-26 08:16:17,017:INFO::EarlyStopping counter: 6 out of 30
2024-12-26 08:16:18,301:INFO::Epoch_batch_00063 | lr 0.0005 |Train_Loss 0.5288 |  total_loss_consistency1 0.0737 | Loss 0.6025 | Val_Loss 0.5792| Train Time(s) 1.1797| Val Time(s) 0.1010 | Time(s) 1.2807
2024-12-26 08:16:18,301:INFO::EarlyStopping counter: 7 out of 30
2024-12-26 08:16:19,358:INFO::Epoch_batch_00064 | lr 0.0005 |Train_Loss 0.5274 |  total_loss_consistency1 0.0725 | Loss 0.5999 | Val_Loss 0.5793| Train Time(s) 0.9805| Val Time(s) 0.0736 | Time(s) 1.0541
2024-12-26 08:16:19,358:INFO::EarlyStopping counter: 8 out of 30
2024-12-26 08:16:20,416:INFO::Epoch_batch_00065 | lr 0.0005 |Train_Loss 0.5282 |  total_loss_consistency1 0.0710 | Loss 0.5993 | Val_Loss 0.5791| Train Time(s) 0.9681| Val Time(s) 0.0867 | Time(s) 1.0548
2024-12-26 08:16:20,416:INFO::EarlyStopping counter: 9 out of 30
2024-12-26 08:16:21,440:INFO::Epoch_batch_00066 | lr 0.0005 |Train_Loss 0.5277 |  total_loss_consistency1 0.0696 | Loss 0.5973 | Val_Loss 0.5793| Train Time(s) 0.9154| Val Time(s) 0.1062 | Time(s) 1.0216
2024-12-26 08:16:21,440:INFO::EarlyStopping counter: 10 out of 30
2024-12-26 08:16:22,741:INFO::Epoch_batch_00067 | lr 0.0005 |Train_Loss 0.5277 |  total_loss_consistency1 0.0684 | Loss 0.5960 | Val_Loss 0.5794| Train Time(s) 1.1934| Val Time(s) 0.1046 | Time(s) 1.2980
2024-12-26 08:16:22,741:INFO::EarlyStopping counter: 11 out of 30
2024-12-26 08:16:23,796:INFO::Epoch_batch_00068 | lr 0.0005 |Train_Loss 0.5269 |  total_loss_consistency1 0.0665 | Loss 0.5934 | Val_Loss 0.5796| Train Time(s) 0.9493| Val Time(s) 0.1022 | Time(s) 1.0515
2024-12-26 08:16:23,796:INFO::EarlyStopping counter: 12 out of 30
2024-12-26 08:16:24,853:INFO::Epoch_batch_00069 | lr 0.0005 |Train_Loss 0.5266 |  total_loss_consistency1 0.0649 | Loss 0.5914 | Val_Loss 0.5793| Train Time(s) 0.9505| Val Time(s) 0.1039 | Time(s) 1.0544
2024-12-26 08:16:24,853:INFO::EarlyStopping counter: 13 out of 30
2024-12-26 08:16:25,914:INFO::Epoch_batch_00070 | lr 0.0005 |Train_Loss 0.5272 |  total_loss_consistency1 0.0630 | Loss 0.5902 | Val_Loss 0.5795| Train Time(s) 0.9091| Val Time(s) 0.1477 | Time(s) 1.0568
2024-12-26 08:16:25,915:INFO::EarlyStopping counter: 14 out of 30
2024-12-26 08:16:27,192:INFO::Epoch_batch_00071 | lr 0.0005 |Train_Loss 0.5268 |  total_loss_consistency1 0.0614 | Loss 0.5882 | Val_Loss 0.5795| Train Time(s) 1.1627| Val Time(s) 0.1118 | Time(s) 1.2745
2024-12-26 08:16:27,192:INFO::EarlyStopping counter: 15 out of 30
2024-12-26 08:16:28,239:INFO::Epoch_batch_00072 | lr 0.0005 |Train_Loss 0.5257 |  total_loss_consistency1 0.0597 | Loss 0.5855 | Val_Loss 0.5797| Train Time(s) 0.9318| Val Time(s) 0.1120 | Time(s) 1.0438
2024-12-26 08:16:28,239:INFO::EarlyStopping counter: 16 out of 30
2024-12-26 08:16:29,286:INFO::Epoch_batch_00073 | lr 0.0005 |Train_Loss 0.5266 |  total_loss_consistency1 0.0578 | Loss 0.5844 | Val_Loss 0.5802| Train Time(s) 0.9312| Val Time(s) 0.1130 | Time(s) 1.0442
2024-12-26 08:16:29,286:INFO::EarlyStopping counter: 17 out of 30
2024-12-26 08:16:30,428:INFO::Epoch_batch_00074 | lr 0.0005 |Train_Loss 0.5264 |  total_loss_consistency1 0.0560 | Loss 0.5824 | Val_Loss 0.5802| Train Time(s) 0.9581| Val Time(s) 0.1787 | Time(s) 1.1368
2024-12-26 08:16:30,428:INFO::EarlyStopping counter: 18 out of 30
2024-12-26 08:16:31,608:INFO::Epoch_batch_00075 | lr 0.0005 |Train_Loss 0.5247 |  total_loss_consistency1 0.0543 | Loss 0.5790 | Val_Loss 0.5802| Train Time(s) 1.0652| Val Time(s) 0.1121 | Time(s) 1.1772
2024-12-26 08:16:31,608:INFO::EarlyStopping counter: 19 out of 30
2024-12-26 08:16:32,679:INFO::Epoch_batch_00076 | lr 0.0005 |Train_Loss 0.5260 |  total_loss_consistency1 0.0523 | Loss 0.5782 | Val_Loss 0.5801| Train Time(s) 0.9429| Val Time(s) 0.1273 | Time(s) 1.0702
2024-12-26 08:16:32,679:INFO::EarlyStopping counter: 20 out of 30
2024-12-26 08:16:33,687:INFO::Epoch_batch_00077 | lr 0.0005 |Train_Loss 0.5249 |  total_loss_consistency1 0.0501 | Loss 0.5750 | Val_Loss 0.5805| Train Time(s) 0.9203| Val Time(s) 0.0876 | Time(s) 1.0079
2024-12-26 08:16:33,688:INFO::EarlyStopping counter: 21 out of 30
2024-12-26 08:16:34,968:INFO::Epoch_batch_00078 | lr 0.0005 |Train_Loss 0.5266 |  total_loss_consistency1 0.0483 | Loss 0.5748 | Val_Loss 0.5806| Train Time(s) 1.1185| Val Time(s) 0.1569 | Time(s) 1.2754
2024-12-26 08:16:34,968:INFO::EarlyStopping counter: 22 out of 30
2024-12-26 08:16:36,035:INFO::Epoch_batch_00079 | lr 0.0005 |Train_Loss 0.5252 |  total_loss_consistency1 0.0463 | Loss 0.5714 | Val_Loss 0.5809| Train Time(s) 0.9856| Val Time(s) 0.0787 | Time(s) 1.0643
2024-12-26 08:16:36,035:INFO::EarlyStopping counter: 23 out of 30
2024-12-26 08:16:37,088:INFO::Epoch_batch_00080 | lr 0.0005 |Train_Loss 0.5249 |  total_loss_consistency1 0.0445 | Loss 0.5695 | Val_Loss 0.5807| Train Time(s) 0.9586| Val Time(s) 0.0915 | Time(s) 1.0500
2024-12-26 08:16:37,089:INFO::EarlyStopping counter: 24 out of 30
2024-12-26 08:16:38,112:INFO::Epoch_batch_00081 | lr 0.0005 |Train_Loss 0.5258 |  total_loss_consistency1 0.0428 | Loss 0.5686 | Val_Loss 0.5808| Train Time(s) 0.9081| Val Time(s) 0.1129 | Time(s) 1.0209
2024-12-26 08:16:38,113:INFO::EarlyStopping counter: 25 out of 30
2024-12-26 08:16:39,418:INFO::Epoch_batch_00082 | lr 0.0005 |Train_Loss 0.5252 |  total_loss_consistency1 0.0407 | Loss 0.5659 | Val_Loss 0.5804| Train Time(s) 1.1982| Val Time(s) 0.1042 | Time(s) 1.3024
2024-12-26 08:16:39,418:INFO::EarlyStopping counter: 26 out of 30
2024-12-26 08:16:40,478:INFO::Epoch_batch_00083 | lr 0.0005 |Train_Loss 0.5254 |  total_loss_consistency1 0.0390 | Loss 0.5644 | Val_Loss 0.5807| Train Time(s) 0.9536| Val Time(s) 0.1038 | Time(s) 1.0573
2024-12-26 08:16:40,479:INFO::EarlyStopping counter: 27 out of 30
2024-12-26 08:16:41,531:INFO::Epoch_batch_00084 | lr 0.0005 |Train_Loss 0.5238 |  total_loss_consistency1 0.0378 | Loss 0.5617 | Val_Loss 0.5811| Train Time(s) 0.9426| Val Time(s) 0.1067 | Time(s) 1.0493
2024-12-26 08:16:41,531:INFO::EarlyStopping counter: 28 out of 30
2024-12-26 08:16:42,557:INFO::Epoch_batch_00085 | lr 0.0005 |Train_Loss 0.5243 |  total_loss_consistency1 0.0361 | Loss 0.5604 | Val_Loss 0.5819| Train Time(s) 0.9111| Val Time(s) 0.1123 | Time(s) 1.0234
2024-12-26 08:16:42,557:INFO::EarlyStopping counter: 29 out of 30
2024-12-26 08:16:43,867:INFO::Epoch_batch_00086 | lr 0.0005 |Train_Loss 0.5248 |  total_loss_consistency1 0.0348 | Loss 0.5596 | Val_Loss 0.5825| Train Time(s) 1.1936| Val Time(s) 0.1136 | Time(s) 1.3073
2024-12-26 08:16:43,868:INFO::EarlyStopping counter: 30 out of 30
2024-12-26 08:16:43,868:INFO::Eearly stopping!
2024-12-26 08:16:43,956:INFO::
testing...
2024-12-26 08:16:44,398:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:16:44,398:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:16:44,398:INFO::submit dir: submit/submit_simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:16:44,504:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:16:44,518:INFO::{'micro-f1': 0.6890167175984181, 'macro-f1': 0.6583317532374553}
2024-12-26 08:16:44,562:INFO::############### Experiments Stage Ends! ###############
2024-12-26 08:16:44,562:INFO::=============== one experiment stage finish, use 117.8707423210144 time.
2024-12-26 08:16:47,827:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:16:49,164:INFO::Epoch_batch_00000 | lr 0.0005 |Train_Loss 0.7316 |  total_loss_consistency1 0.1737 | Loss 0.9053 | Val_Loss 0.6458| Train Time(s) 0.9911| Val Time(s) 0.0798 | Time(s) 1.0709
2024-12-26 08:16:50,416:INFO::Epoch_batch_00001 | lr 0.0005 |Train_Loss 0.6443 |  total_loss_consistency1 0.1635 | Loss 0.8078 | Val_Loss 0.6371| Train Time(s) 0.9360| Val Time(s) 0.0712 | Time(s) 1.0072
2024-12-26 08:16:50,416:INFO::Validation loss decreased (inf --> 0.637121).  Saving model ...
2024-12-26 08:16:51,962:INFO::Epoch_batch_00002 | lr 0.0005 |Train_Loss 0.6347 |  total_loss_consistency1 0.1562 | Loss 0.7909 | Val_Loss 0.6339| Train Time(s) 1.2242| Val Time(s) 0.0818 | Time(s) 1.3060
2024-12-26 08:16:51,962:INFO::Validation loss decreased (0.637121 --> 0.633896).  Saving model ...
2024-12-26 08:16:53,305:INFO::Epoch_batch_00003 | lr 0.0005 |Train_Loss 0.6310 |  total_loss_consistency1 0.1496 | Loss 0.7807 | Val_Loss 0.6321| Train Time(s) 0.9856| Val Time(s) 0.1111 | Time(s) 1.0967
2024-12-26 08:16:53,306:INFO::Validation loss decreased (0.633896 --> 0.632114).  Saving model ...
2024-12-26 08:16:54,577:INFO::Epoch_batch_00004 | lr 0.0005 |Train_Loss 0.6294 |  total_loss_consistency1 0.1450 | Loss 0.7745 | Val_Loss 0.6307| Train Time(s) 0.9252| Val Time(s) 0.1023 | Time(s) 1.0274
2024-12-26 08:16:54,578:INFO::Validation loss decreased (0.632114 --> 0.630680).  Saving model ...
2024-12-26 08:16:56,118:INFO::Epoch_batch_00005 | lr 0.0005 |Train_Loss 0.6282 |  total_loss_consistency1 0.1403 | Loss 0.7685 | Val_Loss 0.6293| Train Time(s) 1.1875| Val Time(s) 0.1127 | Time(s) 1.3003
2024-12-26 08:16:56,118:INFO::Validation loss decreased (0.630680 --> 0.629287).  Saving model ...
2024-12-26 08:16:57,449:INFO::Epoch_batch_00006 | lr 0.0005 |Train_Loss 0.6267 |  total_loss_consistency1 0.1367 | Loss 0.7633 | Val_Loss 0.6279| Train Time(s) 0.9475| Val Time(s) 0.1438 | Time(s) 1.0913
2024-12-26 08:16:57,450:INFO::Validation loss decreased (0.629287 --> 0.627884).  Saving model ...
2024-12-26 08:16:58,700:INFO::Epoch_batch_00007 | lr 0.0005 |Train_Loss 0.6236 |  total_loss_consistency1 0.1334 | Loss 0.7570 | Val_Loss 0.6264| Train Time(s) 0.9011| Val Time(s) 0.1119 | Time(s) 1.0129
2024-12-26 08:16:58,700:INFO::Validation loss decreased (0.627884 --> 0.626397).  Saving model ...
2024-12-26 08:17:00,273:INFO::Epoch_batch_00008 | lr 0.0005 |Train_Loss 0.6236 |  total_loss_consistency1 0.1309 | Loss 0.7545 | Val_Loss 0.6248| Train Time(s) 1.2101| Val Time(s) 0.1130 | Time(s) 1.3231
2024-12-26 08:17:00,273:INFO::Validation loss decreased (0.626397 --> 0.624779).  Saving model ...
2024-12-26 08:17:01,612:INFO::Epoch_batch_00009 | lr 0.0005 |Train_Loss 0.6206 |  total_loss_consistency1 0.1284 | Loss 0.7490 | Val_Loss 0.6230| Train Time(s) 0.9722| Val Time(s) 0.1215 | Time(s) 1.0937
2024-12-26 08:17:01,612:INFO::Validation loss decreased (0.624779 --> 0.623011).  Saving model ...
2024-12-26 08:17:02,873:INFO::Epoch_batch_00010 | lr 0.0005 |Train_Loss 0.6194 |  total_loss_consistency1 0.1260 | Loss 0.7454 | Val_Loss 0.6211| Train Time(s) 0.9246| Val Time(s) 0.0991 | Time(s) 1.0237
2024-12-26 08:17:02,874:INFO::Validation loss decreased (0.623011 --> 0.621079).  Saving model ...
2024-12-26 08:17:04,418:INFO::Epoch_batch_00011 | lr 0.0005 |Train_Loss 0.6180 |  total_loss_consistency1 0.1236 | Loss 0.7416 | Val_Loss 0.6190| Train Time(s) 1.1991| Val Time(s) 0.1049 | Time(s) 1.3040
2024-12-26 08:17:04,418:INFO::Validation loss decreased (0.621079 --> 0.618991).  Saving model ...
2024-12-26 08:17:05,740:INFO::Epoch_batch_00012 | lr 0.0005 |Train_Loss 0.6147 |  total_loss_consistency1 0.1217 | Loss 0.7364 | Val_Loss 0.6167| Train Time(s) 0.9510| Val Time(s) 0.1335 | Time(s) 1.0845
2024-12-26 08:17:05,740:INFO::Validation loss decreased (0.618991 --> 0.616730).  Saving model ...
2024-12-26 08:17:07,005:INFO::Epoch_batch_00013 | lr 0.0005 |Train_Loss 0.6138 |  total_loss_consistency1 0.1200 | Loss 0.7338 | Val_Loss 0.6143| Train Time(s) 0.9216| Val Time(s) 0.1038 | Time(s) 1.0254
2024-12-26 08:17:07,006:INFO::Validation loss decreased (0.616730 --> 0.614281).  Saving model ...
2024-12-26 08:17:08,561:INFO::Epoch_batch_00014 | lr 0.0005 |Train_Loss 0.6108 |  total_loss_consistency1 0.1185 | Loss 0.7293 | Val_Loss 0.6116| Train Time(s) 1.2018| Val Time(s) 0.1125 | Time(s) 1.3144
2024-12-26 08:17:08,562:INFO::Validation loss decreased (0.614281 --> 0.611583).  Saving model ...
2024-12-26 08:17:09,878:INFO::Epoch_batch_00015 | lr 0.0005 |Train_Loss 0.6099 |  total_loss_consistency1 0.1174 | Loss 0.7272 | Val_Loss 0.6087| Train Time(s) 0.9334| Val Time(s) 0.1427 | Time(s) 1.0760
2024-12-26 08:17:09,878:INFO::Validation loss decreased (0.611583 --> 0.608697).  Saving model ...
2024-12-26 08:17:11,129:INFO::Epoch_batch_00016 | lr 0.0005 |Train_Loss 0.6052 |  total_loss_consistency1 0.1161 | Loss 0.7213 | Val_Loss 0.6056| Train Time(s) 0.9043| Val Time(s) 0.1104 | Time(s) 1.0147
2024-12-26 08:17:11,130:INFO::Validation loss decreased (0.608697 --> 0.605577).  Saving model ...
2024-12-26 08:17:12,667:INFO::Epoch_batch_00017 | lr 0.0005 |Train_Loss 0.6013 |  total_loss_consistency1 0.1153 | Loss 0.7166 | Val_Loss 0.6023| Train Time(s) 1.1837| Val Time(s) 0.1137 | Time(s) 1.2974
2024-12-26 08:17:12,668:INFO::Validation loss decreased (0.605577 --> 0.602271).  Saving model ...
2024-12-26 08:17:13,964:INFO::Epoch_batch_00018 | lr 0.0005 |Train_Loss 0.5972 |  total_loss_consistency1 0.1145 | Loss 0.7117 | Val_Loss 0.5988| Train Time(s) 0.9389| Val Time(s) 0.1119 | Time(s) 1.0509
2024-12-26 08:17:13,964:INFO::Validation loss decreased (0.602271 --> 0.598829).  Saving model ...
2024-12-26 08:17:15,215:INFO::Epoch_batch_00019 | lr 0.0005 |Train_Loss 0.5937 |  total_loss_consistency1 0.1136 | Loss 0.7073 | Val_Loss 0.5954| Train Time(s) 0.8990| Val Time(s) 0.1126 | Time(s) 1.0116
2024-12-26 08:17:15,216:INFO::Validation loss decreased (0.598829 --> 0.595370).  Saving model ...
2024-12-26 08:17:16,746:INFO::Epoch_batch_00020 | lr 0.0005 |Train_Loss 0.5911 |  total_loss_consistency1 0.1128 | Loss 0.7039 | Val_Loss 0.5921| Train Time(s) 1.1931| Val Time(s) 0.1025 | Time(s) 1.2955
2024-12-26 08:17:16,746:INFO::Validation loss decreased (0.595370 --> 0.592108).  Saving model ...
2024-12-26 08:17:18,024:INFO::Epoch_batch_00021 | lr 0.0005 |Train_Loss 0.5859 |  total_loss_consistency1 0.1123 | Loss 0.6982 | Val_Loss 0.5891| Train Time(s) 0.9514| Val Time(s) 0.0879 | Time(s) 1.0393
2024-12-26 08:17:18,024:INFO::Validation loss decreased (0.592108 --> 0.589079).  Saving model ...
2024-12-26 08:17:19,287:INFO::Epoch_batch_00022 | lr 0.0005 |Train_Loss 0.5811 |  total_loss_consistency1 0.1115 | Loss 0.6926 | Val_Loss 0.5865| Train Time(s) 0.9504| Val Time(s) 0.0763 | Time(s) 1.0267
2024-12-26 08:17:19,288:INFO::Validation loss decreased (0.589079 --> 0.586456).  Saving model ...
2024-12-26 08:17:20,833:INFO::Epoch_batch_00023 | lr 0.0005 |Train_Loss 0.5797 |  total_loss_consistency1 0.1109 | Loss 0.6906 | Val_Loss 0.5847| Train Time(s) 1.2242| Val Time(s) 0.0812 | Time(s) 1.3054
2024-12-26 08:17:20,834:INFO::Validation loss decreased (0.586456 --> 0.584657).  Saving model ...
2024-12-26 08:17:22,124:INFO::Epoch_batch_00024 | lr 0.0005 |Train_Loss 0.5768 |  total_loss_consistency1 0.1101 | Loss 0.6869 | Val_Loss 0.5834| Train Time(s) 0.9546| Val Time(s) 0.0989 | Time(s) 1.0535
2024-12-26 08:17:22,124:INFO::Validation loss decreased (0.584657 --> 0.583403).  Saving model ...
2024-12-26 08:17:23,411:INFO::Epoch_batch_00025 | lr 0.0005 |Train_Loss 0.5754 |  total_loss_consistency1 0.1091 | Loss 0.6845 | Val_Loss 0.5825| Train Time(s) 0.9346| Val Time(s) 0.1124 | Time(s) 1.0470
2024-12-26 08:17:23,412:INFO::Validation loss decreased (0.583403 --> 0.582455).  Saving model ...
2024-12-26 08:17:24,956:INFO::Epoch_batch_00026 | lr 0.0005 |Train_Loss 0.5702 |  total_loss_consistency1 0.1082 | Loss 0.6784 | Val_Loss 0.5815| Train Time(s) 1.1419| Val Time(s) 0.1607 | Time(s) 1.3026
2024-12-26 08:17:24,957:INFO::Validation loss decreased (0.582455 --> 0.581504).  Saving model ...
2024-12-26 08:17:26,261:INFO::Epoch_batch_00027 | lr 0.0005 |Train_Loss 0.5678 |  total_loss_consistency1 0.1069 | Loss 0.6748 | Val_Loss 0.5808| Train Time(s) 0.9570| Val Time(s) 0.1082 | Time(s) 1.0652
2024-12-26 08:17:26,262:INFO::Validation loss decreased (0.581504 --> 0.580806).  Saving model ...
2024-12-26 08:17:27,558:INFO::Epoch_batch_00028 | lr 0.0005 |Train_Loss 0.5669 |  total_loss_consistency1 0.1060 | Loss 0.6729 | Val_Loss 0.5802| Train Time(s) 0.9471| Val Time(s) 0.1127 | Time(s) 1.0598
2024-12-26 08:17:27,559:INFO::Validation loss decreased (0.580806 --> 0.580235).  Saving model ...
2024-12-26 08:17:29,108:INFO::Epoch_batch_00029 | lr 0.0005 |Train_Loss 0.5639 |  total_loss_consistency1 0.1049 | Loss 0.6688 | Val_Loss 0.5798| Train Time(s) 1.1838| Val Time(s) 0.1203 | Time(s) 1.3041
2024-12-26 08:17:29,109:INFO::Validation loss decreased (0.580235 --> 0.579812).  Saving model ...
2024-12-26 08:17:30,419:INFO::Epoch_batch_00030 | lr 0.0005 |Train_Loss 0.5618 |  total_loss_consistency1 0.1041 | Loss 0.6658 | Val_Loss 0.5798| Train Time(s) 0.9557| Val Time(s) 0.1119 | Time(s) 1.0676
2024-12-26 08:17:30,419:INFO::Validation loss decreased (0.579812 --> 0.579753).  Saving model ...
2024-12-26 08:17:31,590:INFO::Epoch_batch_00031 | lr 0.0005 |Train_Loss 0.5610 |  total_loss_consistency1 0.1032 | Loss 0.6642 | Val_Loss 0.5792| Train Time(s) 0.8863| Val Time(s) 0.0498 | Time(s) 0.9361
2024-12-26 08:17:31,590:INFO::Validation loss decreased (0.579753 --> 0.579210).  Saving model ...
2024-12-26 08:17:32,704:INFO::Epoch_batch_00032 | lr 0.0005 |Train_Loss 0.5600 |  total_loss_consistency1 0.1024 | Loss 0.6623 | Val_Loss 0.5790| Train Time(s) 0.7592| Val Time(s) 0.1175 | Time(s) 0.8766
2024-12-26 08:17:32,704:INFO::Validation loss decreased (0.579210 --> 0.578957).  Saving model ...
2024-12-26 08:17:33,723:INFO::Epoch_batch_00033 | lr 0.0005 |Train_Loss 0.5569 |  total_loss_consistency1 0.1014 | Loss 0.6584 | Val_Loss 0.5788| Train Time(s) 0.7326| Val Time(s) 0.0499 | Time(s) 0.7825
2024-12-26 08:17:33,723:INFO::Validation loss decreased (0.578957 --> 0.578827).  Saving model ...
2024-12-26 08:17:34,765:INFO::Epoch_batch_00034 | lr 0.0005 |Train_Loss 0.5557 |  total_loss_consistency1 0.1006 | Loss 0.6563 | Val_Loss 0.5787| Train Time(s) 0.7552| Val Time(s) 0.0518 | Time(s) 0.8070
2024-12-26 08:17:34,766:INFO::Validation loss decreased (0.578827 --> 0.578699).  Saving model ...
2024-12-26 08:17:35,510:INFO::Epoch_batch_00035 | lr 0.0005 |Train_Loss 0.5534 |  total_loss_consistency1 0.0997 | Loss 0.6531 | Val_Loss 0.5787| Train Time(s) 0.6940| Val Time(s) 0.0501 | Time(s) 0.7441
2024-12-26 08:17:35,510:INFO::EarlyStopping counter: 1 out of 30
2024-12-26 08:17:36,498:INFO::Epoch_batch_00036 | lr 0.0005 |Train_Loss 0.5519 |  total_loss_consistency1 0.0987 | Loss 0.6506 | Val_Loss 0.5789| Train Time(s) 0.8432| Val Time(s) 0.1416 | Time(s) 0.9848
2024-12-26 08:17:36,498:INFO::EarlyStopping counter: 2 out of 30
2024-12-26 08:17:37,315:INFO::Epoch_batch_00037 | lr 0.0005 |Train_Loss 0.5506 |  total_loss_consistency1 0.0982 | Loss 0.6488 | Val_Loss 0.5788| Train Time(s) 0.7323| Val Time(s) 0.0816 | Time(s) 0.8139
2024-12-26 08:17:37,315:INFO::EarlyStopping counter: 3 out of 30
2024-12-26 08:17:38,094:INFO::Epoch_batch_00038 | lr 0.0005 |Train_Loss 0.5490 |  total_loss_consistency1 0.0974 | Loss 0.6464 | Val_Loss 0.5787| Train Time(s) 0.7276| Val Time(s) 0.0502 | Time(s) 0.7778
2024-12-26 08:17:38,094:INFO::EarlyStopping counter: 4 out of 30
2024-12-26 08:17:38,909:INFO::Epoch_batch_00039 | lr 0.0005 |Train_Loss 0.5477 |  total_loss_consistency1 0.0965 | Loss 0.6442 | Val_Loss 0.5788| Train Time(s) 0.7442| Val Time(s) 0.0707 | Time(s) 0.8149
2024-12-26 08:17:38,910:INFO::EarlyStopping counter: 5 out of 30
2024-12-26 08:17:39,692:INFO::Epoch_batch_00040 | lr 0.0005 |Train_Loss 0.5457 |  total_loss_consistency1 0.0956 | Loss 0.6413 | Val_Loss 0.5790| Train Time(s) 0.6982| Val Time(s) 0.0839 | Time(s) 0.7820
2024-12-26 08:17:39,692:INFO::EarlyStopping counter: 6 out of 30
2024-12-26 08:17:40,743:INFO::Epoch_batch_00041 | lr 0.0005 |Train_Loss 0.5460 |  total_loss_consistency1 0.0948 | Loss 0.6408 | Val_Loss 0.5790| Train Time(s) 0.9443| Val Time(s) 0.1059 | Time(s) 1.0503
2024-12-26 08:17:40,744:INFO::EarlyStopping counter: 7 out of 30
2024-12-26 08:17:41,530:INFO::Epoch_batch_00042 | lr 0.0005 |Train_Loss 0.5435 |  total_loss_consistency1 0.0942 | Loss 0.6377 | Val_Loss 0.5791| Train Time(s) 0.6959| Val Time(s) 0.0901 | Time(s) 0.7860
2024-12-26 08:17:41,530:INFO::EarlyStopping counter: 8 out of 30
2024-12-26 08:17:42,346:INFO::Epoch_batch_00043 | lr 0.0005 |Train_Loss 0.5433 |  total_loss_consistency1 0.0935 | Loss 0.6367 | Val_Loss 0.5791| Train Time(s) 0.7317| Val Time(s) 0.0830 | Time(s) 0.8148
2024-12-26 08:17:42,346:INFO::EarlyStopping counter: 9 out of 30
2024-12-26 08:17:43,158:INFO::Epoch_batch_00044 | lr 0.0005 |Train_Loss 0.5432 |  total_loss_consistency1 0.0926 | Loss 0.6358 | Val_Loss 0.5792| Train Time(s) 0.7282| Val Time(s) 0.0837 | Time(s) 0.8119
2024-12-26 08:17:43,159:INFO::EarlyStopping counter: 10 out of 30
2024-12-26 08:17:43,939:INFO::Epoch_batch_00045 | lr 0.0005 |Train_Loss 0.5420 |  total_loss_consistency1 0.0917 | Loss 0.6337 | Val_Loss 0.5792| Train Time(s) 0.6928| Val Time(s) 0.0871 | Time(s) 0.7799
2024-12-26 08:17:43,940:INFO::EarlyStopping counter: 11 out of 30
2024-12-26 08:17:44,993:INFO::Epoch_batch_00046 | lr 0.0005 |Train_Loss 0.5391 |  total_loss_consistency1 0.0909 | Loss 0.6300 | Val_Loss 0.5793| Train Time(s) 0.9814| Val Time(s) 0.0717 | Time(s) 1.0532
2024-12-26 08:17:44,993:INFO::EarlyStopping counter: 12 out of 30
2024-12-26 08:17:45,809:INFO::Epoch_batch_00047 | lr 0.0005 |Train_Loss 0.5386 |  total_loss_consistency1 0.0901 | Loss 0.6287 | Val_Loss 0.5795| Train Time(s) 0.7334| Val Time(s) 0.0823 | Time(s) 0.8157
2024-12-26 08:17:45,810:INFO::EarlyStopping counter: 13 out of 30
2024-12-26 08:17:46,620:INFO::Epoch_batch_00048 | lr 0.0005 |Train_Loss 0.5386 |  total_loss_consistency1 0.0893 | Loss 0.6279 | Val_Loss 0.5796| Train Time(s) 0.7071| Val Time(s) 0.1023 | Time(s) 0.8095
2024-12-26 08:17:46,620:INFO::EarlyStopping counter: 14 out of 30
2024-12-26 08:17:47,403:INFO::Epoch_batch_00049 | lr 0.0005 |Train_Loss 0.5382 |  total_loss_consistency1 0.0885 | Loss 0.6266 | Val_Loss 0.5795| Train Time(s) 0.6981| Val Time(s) 0.0840 | Time(s) 0.7821
2024-12-26 08:17:47,403:INFO::EarlyStopping counter: 15 out of 30
2024-12-26 08:17:48,211:INFO::Epoch_batch_00050 | lr 0.0005 |Train_Loss 0.5354 |  total_loss_consistency1 0.0877 | Loss 0.6230 | Val_Loss 0.5797| Train Time(s) 0.6930| Val Time(s) 0.1118 | Time(s) 0.8048
2024-12-26 08:17:48,211:INFO::EarlyStopping counter: 16 out of 30
2024-12-26 08:17:49,204:INFO::Epoch_batch_00051 | lr 0.0005 |Train_Loss 0.5353 |  total_loss_consistency1 0.0865 | Loss 0.6218 | Val_Loss 0.5801| Train Time(s) 0.9101| Val Time(s) 0.0799 | Time(s) 0.9899
2024-12-26 08:17:49,204:INFO::EarlyStopping counter: 17 out of 30
2024-12-26 08:17:49,986:INFO::Epoch_batch_00052 | lr 0.0005 |Train_Loss 0.5368 |  total_loss_consistency1 0.0857 | Loss 0.6225 | Val_Loss 0.5800| Train Time(s) 0.7311| Val Time(s) 0.0502 | Time(s) 0.7813
2024-12-26 08:17:49,986:INFO::EarlyStopping counter: 18 out of 30
2024-12-26 08:17:50,799:INFO::Epoch_batch_00053 | lr 0.0005 |Train_Loss 0.5356 |  total_loss_consistency1 0.0848 | Loss 0.6204 | Val_Loss 0.5799| Train Time(s) 0.7272| Val Time(s) 0.0855 | Time(s) 0.8127
2024-12-26 08:17:50,799:INFO::EarlyStopping counter: 19 out of 30
2024-12-26 08:17:51,579:INFO::Epoch_batch_00054 | lr 0.0005 |Train_Loss 0.5338 |  total_loss_consistency1 0.0840 | Loss 0.6177 | Val_Loss 0.5798| Train Time(s) 0.6911| Val Time(s) 0.0877 | Time(s) 0.7788
2024-12-26 08:17:51,579:INFO::EarlyStopping counter: 20 out of 30
2024-12-26 08:17:52,562:INFO::Epoch_batch_00055 | lr 0.0005 |Train_Loss 0.5329 |  total_loss_consistency1 0.0831 | Loss 0.6160 | Val_Loss 0.5799| Train Time(s) 0.8447| Val Time(s) 0.1356 | Time(s) 0.9802
2024-12-26 08:17:52,562:INFO::EarlyStopping counter: 21 out of 30
2024-12-26 08:17:53,392:INFO::Epoch_batch_00056 | lr 0.0005 |Train_Loss 0.5325 |  total_loss_consistency1 0.0820 | Loss 0.6145 | Val_Loss 0.5803| Train Time(s) 0.7406| Val Time(s) 0.0865 | Time(s) 0.8272
2024-12-26 08:17:53,393:INFO::EarlyStopping counter: 22 out of 30
2024-12-26 08:17:54,163:INFO::Epoch_batch_00057 | lr 0.0005 |Train_Loss 0.5332 |  total_loss_consistency1 0.0808 | Loss 0.6140 | Val_Loss 0.5801| Train Time(s) 0.7206| Val Time(s) 0.0500 | Time(s) 0.7706
2024-12-26 08:17:54,164:INFO::EarlyStopping counter: 23 out of 30
2024-12-26 08:17:54,977:INFO::Epoch_batch_00058 | lr 0.0005 |Train_Loss 0.5330 |  total_loss_consistency1 0.0796 | Loss 0.6126 | Val_Loss 0.5800| Train Time(s) 0.7335| Val Time(s) 0.0795 | Time(s) 0.8130
2024-12-26 08:17:54,978:INFO::EarlyStopping counter: 24 out of 30
2024-12-26 08:17:55,759:INFO::Epoch_batch_00059 | lr 0.0005 |Train_Loss 0.5311 |  total_loss_consistency1 0.0785 | Loss 0.6095 | Val_Loss 0.5800| Train Time(s) 0.6952| Val Time(s) 0.0861 | Time(s) 0.7813
2024-12-26 08:17:55,760:INFO::EarlyStopping counter: 25 out of 30
2024-12-26 08:17:56,814:INFO::Epoch_batch_00060 | lr 0.0005 |Train_Loss 0.5310 |  total_loss_consistency1 0.0773 | Loss 0.6083 | Val_Loss 0.5802| Train Time(s) 0.9298| Val Time(s) 0.1239 | Time(s) 1.0537
2024-12-26 08:17:56,814:INFO::EarlyStopping counter: 26 out of 30
2024-12-26 08:17:57,605:INFO::Epoch_batch_00061 | lr 0.0005 |Train_Loss 0.5303 |  total_loss_consistency1 0.0760 | Loss 0.6062 | Val_Loss 0.5803| Train Time(s) 0.7130| Val Time(s) 0.0769 | Time(s) 0.7900
2024-12-26 08:17:57,605:INFO::EarlyStopping counter: 27 out of 30
2024-12-26 08:17:58,417:INFO::Epoch_batch_00062 | lr 0.0005 |Train_Loss 0.5299 |  total_loss_consistency1 0.0748 | Loss 0.6047 | Val_Loss 0.5803| Train Time(s) 0.7289| Val Time(s) 0.0833 | Time(s) 0.8122
2024-12-26 08:17:58,418:INFO::EarlyStopping counter: 28 out of 30
2024-12-26 08:17:59,231:INFO::Epoch_batch_00063 | lr 0.0005 |Train_Loss 0.5293 |  total_loss_consistency1 0.0733 | Loss 0.6026 | Val_Loss 0.5802| Train Time(s) 0.7307| Val Time(s) 0.0815 | Time(s) 0.8122
2024-12-26 08:17:59,231:INFO::EarlyStopping counter: 29 out of 30
2024-12-26 08:18:00,013:INFO::Epoch_batch_00064 | lr 0.0005 |Train_Loss 0.5292 |  total_loss_consistency1 0.0720 | Loss 0.6012 | Val_Loss 0.5802| Train Time(s) 0.6978| Val Time(s) 0.0841 | Time(s) 0.7819
2024-12-26 08:18:00,014:INFO::EarlyStopping counter: 30 out of 30
2024-12-26 08:18:00,014:INFO::Eearly stopping!
2024-12-26 08:18:00,098:INFO::
testing...
2024-12-26 08:18:00,383:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:18:00,383:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:18:00,383:INFO::submit dir: submit/submit_simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:18:00,493:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:18:00,508:INFO::{'micro-f1': 0.6886419979799835, 'macro-f1': 0.6479432675466719}
2024-12-26 08:18:00,552:INFO::############### Experiments Stage Ends! ###############
2024-12-26 08:18:00,552:INFO::=============== one experiment stage finish, use 193.8606140613556 time.
2024-12-26 08:18:03,826:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:18:05,139:INFO::Epoch_batch_00000 | lr 0.0005 |Train_Loss 0.7568 |  total_loss_consistency1 0.1724 | Loss 0.9292 | Val_Loss 0.6471| Train Time(s) 0.9717| Val Time(s) 0.0502 | Time(s) 1.0219
2024-12-26 08:18:06,190:INFO::Epoch_batch_00001 | lr 0.0005 |Train_Loss 0.6534 |  total_loss_consistency1 0.1682 | Loss 0.8216 | Val_Loss 0.6376| Train Time(s) 0.7202| Val Time(s) 0.0920 | Time(s) 0.8122
2024-12-26 08:18:06,190:INFO::Validation loss decreased (inf --> 0.637561).  Saving model ...
2024-12-26 08:18:07,223:INFO::Epoch_batch_00002 | lr 0.0005 |Train_Loss 0.6369 |  total_loss_consistency1 0.1630 | Loss 0.7999 | Val_Loss 0.6350| Train Time(s) 0.7502| Val Time(s) 0.0499 | Time(s) 0.8000
2024-12-26 08:18:07,224:INFO::Validation loss decreased (0.637561 --> 0.635041).  Saving model ...
2024-12-26 08:18:08,209:INFO::Epoch_batch_00003 | lr 0.0005 |Train_Loss 0.6334 |  total_loss_consistency1 0.1562 | Loss 0.7897 | Val_Loss 0.6338| Train Time(s) 0.7013| Val Time(s) 0.0503 | Time(s) 0.7515
2024-12-26 08:18:08,209:INFO::Validation loss decreased (0.635041 --> 0.633767).  Saving model ...
2024-12-26 08:18:09,461:INFO::Epoch_batch_00004 | lr 0.0005 |Train_Loss 0.6315 |  total_loss_consistency1 0.1497 | Loss 0.7812 | Val_Loss 0.6327| Train Time(s) 0.9615| Val Time(s) 0.0552 | Time(s) 1.0167
2024-12-26 08:18:09,461:INFO::Validation loss decreased (0.633767 --> 0.632692).  Saving model ...
2024-12-26 08:18:10,470:INFO::Epoch_batch_00005 | lr 0.0005 |Train_Loss 0.6297 |  total_loss_consistency1 0.1437 | Loss 0.7734 | Val_Loss 0.6316| Train Time(s) 0.7244| Val Time(s) 0.0504 | Time(s) 0.7749
2024-12-26 08:18:10,470:INFO::Validation loss decreased (0.632692 --> 0.631572).  Saving model ...
2024-12-26 08:18:11,481:INFO::Epoch_batch_00006 | lr 0.0005 |Train_Loss 0.6285 |  total_loss_consistency1 0.1382 | Loss 0.7667 | Val_Loss 0.6303| Train Time(s) 0.7278| Val Time(s) 0.0501 | Time(s) 0.7778
2024-12-26 08:18:11,482:INFO::Validation loss decreased (0.631572 --> 0.630337).  Saving model ...
2024-12-26 08:18:12,565:INFO::Epoch_batch_00007 | lr 0.0005 |Train_Loss 0.6273 |  total_loss_consistency1 0.1342 | Loss 0.7615 | Val_Loss 0.6289| Train Time(s) 0.7348| Val Time(s) 0.1114 | Time(s) 0.8462
2024-12-26 08:18:12,566:INFO::Validation loss decreased (0.630337 --> 0.628892).  Saving model ...
2024-12-26 08:18:13,556:INFO::Epoch_batch_00008 | lr 0.0005 |Train_Loss 0.6262 |  total_loss_consistency1 0.1308 | Loss 0.7570 | Val_Loss 0.6272| Train Time(s) 0.6785| Val Time(s) 0.0765 | Time(s) 0.7549
2024-12-26 08:18:13,557:INFO::Validation loss decreased (0.628892 --> 0.627199).  Saving model ...
2024-12-26 08:18:14,527:INFO::Epoch_batch_00009 | lr 0.0005 |Train_Loss 0.6259 |  total_loss_consistency1 0.1278 | Loss 0.7537 | Val_Loss 0.6254| Train Time(s) 0.6783| Val Time(s) 0.0575 | Time(s) 0.7359
2024-12-26 08:18:14,528:INFO::Validation loss decreased (0.627199 --> 0.625383).  Saving model ...
2024-12-26 08:18:15,515:INFO::Epoch_batch_00010 | lr 0.0005 |Train_Loss 0.6224 |  total_loss_consistency1 0.1257 | Loss 0.7481 | Val_Loss 0.6235| Train Time(s) 0.6989| Val Time(s) 0.0495 | Time(s) 0.7484
2024-12-26 08:18:15,516:INFO::Validation loss decreased (0.625383 --> 0.623456).  Saving model ...
2024-12-26 08:18:16,716:INFO::Epoch_batch_00011 | lr 0.0005 |Train_Loss 0.6212 |  total_loss_consistency1 0.1239 | Loss 0.7451 | Val_Loss 0.6215| Train Time(s) 0.8312| Val Time(s) 0.1162 | Time(s) 0.9474
2024-12-26 08:18:16,716:INFO::Validation loss decreased (0.623456 --> 0.621485).  Saving model ...
2024-12-26 08:18:17,734:INFO::Epoch_batch_00012 | lr 0.0005 |Train_Loss 0.6178 |  total_loss_consistency1 0.1222 | Loss 0.7400 | Val_Loss 0.6195| Train Time(s) 0.7255| Val Time(s) 0.0498 | Time(s) 0.7753
2024-12-26 08:18:17,735:INFO::Validation loss decreased (0.621485 --> 0.619462).  Saving model ...
2024-12-26 08:18:18,779:INFO::Epoch_batch_00013 | lr 0.0005 |Train_Loss 0.6165 |  total_loss_consistency1 0.1207 | Loss 0.7372 | Val_Loss 0.6173| Train Time(s) 0.7502| Val Time(s) 0.0500 | Time(s) 0.8002
2024-12-26 08:18:18,779:INFO::Validation loss decreased (0.619462 --> 0.617338).  Saving model ...
2024-12-26 08:18:19,752:INFO::Epoch_batch_00014 | lr 0.0005 |Train_Loss 0.6162 |  total_loss_consistency1 0.1192 | Loss 0.7354 | Val_Loss 0.6150| Train Time(s) 0.6849| Val Time(s) 0.0491 | Time(s) 0.7341
2024-12-26 08:18:19,753:INFO::Validation loss decreased (0.617338 --> 0.615039).  Saving model ...
2024-12-26 08:18:21,002:INFO::Epoch_batch_00015 | lr 0.0005 |Train_Loss 0.6138 |  total_loss_consistency1 0.1179 | Loss 0.7317 | Val_Loss 0.6125| Train Time(s) 0.9553| Val Time(s) 0.0496 | Time(s) 1.0049
2024-12-26 08:18:21,002:INFO::Validation loss decreased (0.615039 --> 0.612505).  Saving model ...
2024-12-26 08:18:22,041:INFO::Epoch_batch_00016 | lr 0.0005 |Train_Loss 0.6110 |  total_loss_consistency1 0.1168 | Loss 0.7278 | Val_Loss 0.6097| Train Time(s) 0.7148| Val Time(s) 0.0790 | Time(s) 0.7938
2024-12-26 08:18:22,041:INFO::Validation loss decreased (0.612505 --> 0.609709).  Saving model ...
2024-12-26 08:18:23,091:INFO::Epoch_batch_00017 | lr 0.0005 |Train_Loss 0.6087 |  total_loss_consistency1 0.1157 | Loss 0.7244 | Val_Loss 0.6067| Train Time(s) 0.7581| Val Time(s) 0.0500 | Time(s) 0.8081
2024-12-26 08:18:23,092:INFO::Validation loss decreased (0.609709 --> 0.606662).  Saving model ...
2024-12-26 08:18:24,069:INFO::Epoch_batch_00018 | lr 0.0005 |Train_Loss 0.6045 |  total_loss_consistency1 0.1149 | Loss 0.7194 | Val_Loss 0.6035| Train Time(s) 0.6899| Val Time(s) 0.0494 | Time(s) 0.7393
2024-12-26 08:18:24,070:INFO::Validation loss decreased (0.606662 --> 0.603467).  Saving model ...
2024-12-26 08:18:25,331:INFO::Epoch_batch_00019 | lr 0.0005 |Train_Loss 0.6019 |  total_loss_consistency1 0.1136 | Loss 0.7155 | Val_Loss 0.6001| Train Time(s) 0.9660| Val Time(s) 0.0518 | Time(s) 1.0177
2024-12-26 08:18:25,331:INFO::Validation loss decreased (0.603467 --> 0.600062).  Saving model ...
2024-12-26 08:18:26,333:INFO::Epoch_batch_00020 | lr 0.0005 |Train_Loss 0.5985 |  total_loss_consistency1 0.1128 | Loss 0.7114 | Val_Loss 0.5964| Train Time(s) 0.7158| Val Time(s) 0.0496 | Time(s) 0.7653
2024-12-26 08:18:26,334:INFO::Validation loss decreased (0.600062 --> 0.596437).  Saving model ...
2024-12-26 08:18:27,338:INFO::Epoch_batch_00021 | lr 0.0005 |Train_Loss 0.5952 |  total_loss_consistency1 0.1122 | Loss 0.7074 | Val_Loss 0.5929| Train Time(s) 0.7207| Val Time(s) 0.0492 | Time(s) 0.7699
2024-12-26 08:18:27,338:INFO::Validation loss decreased (0.596437 --> 0.592936).  Saving model ...
2024-12-26 08:18:28,388:INFO::Epoch_batch_00022 | lr 0.0005 |Train_Loss 0.5906 |  total_loss_consistency1 0.1115 | Loss 0.7022 | Val_Loss 0.5896| Train Time(s) 0.7086| Val Time(s) 0.0981 | Time(s) 0.8068
2024-12-26 08:18:28,388:INFO::Validation loss decreased (0.592936 --> 0.589580).  Saving model ...
2024-12-26 08:18:29,422:INFO::Epoch_batch_00023 | lr 0.0005 |Train_Loss 0.5865 |  total_loss_consistency1 0.1113 | Loss 0.6978 | Val_Loss 0.5867| Train Time(s) 0.7118| Val Time(s) 0.0817 | Time(s) 0.7935
2024-12-26 08:18:29,422:INFO::Validation loss decreased (0.589580 --> 0.586730).  Saving model ...
2024-12-26 08:18:30,379:INFO::Epoch_batch_00024 | lr 0.0005 |Train_Loss 0.5825 |  total_loss_consistency1 0.1109 | Loss 0.6934 | Val_Loss 0.5846| Train Time(s) 0.6697| Val Time(s) 0.0478 | Time(s) 0.7174
2024-12-26 08:18:30,380:INFO::Validation loss decreased (0.586730 --> 0.584646).  Saving model ...
2024-12-26 08:18:31,369:INFO::Epoch_batch_00025 | lr 0.0005 |Train_Loss 0.5775 |  total_loss_consistency1 0.1108 | Loss 0.6883 | Val_Loss 0.5834| Train Time(s) 0.7019| Val Time(s) 0.0501 | Time(s) 0.7520
2024-12-26 08:18:31,370:INFO::Validation loss decreased (0.584646 --> 0.583412).  Saving model ...
2024-12-26 08:18:32,591:INFO::Epoch_batch_00026 | lr 0.0005 |Train_Loss 0.5743 |  total_loss_consistency1 0.1103 | Loss 0.6846 | Val_Loss 0.5828| Train Time(s) 0.8633| Val Time(s) 0.1162 | Time(s) 0.9795
2024-12-26 08:18:32,591:INFO::Validation loss decreased (0.583412 --> 0.582840).  Saving model ...
2024-12-26 08:18:33,650:INFO::Epoch_batch_00027 | lr 0.0005 |Train_Loss 0.5743 |  total_loss_consistency1 0.1093 | Loss 0.6836 | Val_Loss 0.5825| Train Time(s) 0.7400| Val Time(s) 0.0722 | Time(s) 0.8122
2024-12-26 08:18:33,650:INFO::Validation loss decreased (0.582840 --> 0.582509).  Saving model ...
2024-12-26 08:18:34,700:INFO::Epoch_batch_00028 | lr 0.0005 |Train_Loss 0.5720 |  total_loss_consistency1 0.1086 | Loss 0.6805 | Val_Loss 0.5820| Train Time(s) 0.7617| Val Time(s) 0.0503 | Time(s) 0.8120
2024-12-26 08:18:34,700:INFO::Validation loss decreased (0.582509 --> 0.581993).  Saving model ...
2024-12-26 08:18:35,682:INFO::Epoch_batch_00029 | lr 0.0005 |Train_Loss 0.5673 |  total_loss_consistency1 0.1078 | Loss 0.6750 | Val_Loss 0.5813| Train Time(s) 0.6928| Val Time(s) 0.0497 | Time(s) 0.7425
2024-12-26 08:18:35,683:INFO::Validation loss decreased (0.581993 --> 0.581317).  Saving model ...
2024-12-26 08:18:36,934:INFO::Epoch_batch_00030 | lr 0.0005 |Train_Loss 0.5673 |  total_loss_consistency1 0.1070 | Loss 0.6742 | Val_Loss 0.5806| Train Time(s) 0.9656| Val Time(s) 0.0514 | Time(s) 1.0170
2024-12-26 08:18:36,935:INFO::Validation loss decreased (0.581317 --> 0.580617).  Saving model ...
2024-12-26 08:18:37,950:INFO::Epoch_batch_00031 | lr 0.0005 |Train_Loss 0.5636 |  total_loss_consistency1 0.1063 | Loss 0.6698 | Val_Loss 0.5801| Train Time(s) 0.7241| Val Time(s) 0.0497 | Time(s) 0.7738
2024-12-26 08:18:37,950:INFO::Validation loss decreased (0.580617 --> 0.580122).  Saving model ...
2024-12-26 08:18:38,958:INFO::Epoch_batch_00032 | lr 0.0005 |Train_Loss 0.5621 |  total_loss_consistency1 0.1056 | Loss 0.6677 | Val_Loss 0.5795| Train Time(s) 0.7162| Val Time(s) 0.0491 | Time(s) 0.7653
2024-12-26 08:18:38,958:INFO::Validation loss decreased (0.580122 --> 0.579550).  Saving model ...
2024-12-26 08:18:39,931:INFO::Epoch_batch_00033 | lr 0.0005 |Train_Loss 0.5591 |  total_loss_consistency1 0.1047 | Loss 0.6638 | Val_Loss 0.5793| Train Time(s) 0.6868| Val Time(s) 0.0495 | Time(s) 0.7363
2024-12-26 08:18:39,932:INFO::Validation loss decreased (0.579550 --> 0.579309).  Saving model ...
2024-12-26 08:18:41,190:INFO::Epoch_batch_00034 | lr 0.0005 |Train_Loss 0.5602 |  total_loss_consistency1 0.1035 | Loss 0.6637 | Val_Loss 0.5792| Train Time(s) 0.9631| Val Time(s) 0.0514 | Time(s) 1.0145
2024-12-26 08:18:41,190:INFO::Validation loss decreased (0.579309 --> 0.579157).  Saving model ...
2024-12-26 08:18:42,197:INFO::Epoch_batch_00035 | lr 0.0005 |Train_Loss 0.5553 |  total_loss_consistency1 0.1024 | Loss 0.6577 | Val_Loss 0.5791| Train Time(s) 0.7175| Val Time(s) 0.0487 | Time(s) 0.7662
2024-12-26 08:18:42,198:INFO::Validation loss decreased (0.579157 --> 0.579079).  Saving model ...
2024-12-26 08:18:43,205:INFO::Epoch_batch_00036 | lr 0.0005 |Train_Loss 0.5542 |  total_loss_consistency1 0.1017 | Loss 0.6559 | Val_Loss 0.5790| Train Time(s) 0.7167| Val Time(s) 0.0502 | Time(s) 0.7669
2024-12-26 08:18:43,206:INFO::Validation loss decreased (0.579079 --> 0.579019).  Saving model ...
2024-12-26 08:18:44,322:INFO::Epoch_batch_00037 | lr 0.0005 |Train_Loss 0.5528 |  total_loss_consistency1 0.1008 | Loss 0.6536 | Val_Loss 0.5788| Train Time(s) 0.7579| Val Time(s) 0.1174 | Time(s) 0.8753
2024-12-26 08:18:44,323:INFO::Validation loss decreased (0.579019 --> 0.578797).  Saving model ...
2024-12-26 08:18:45,364:INFO::Epoch_batch_00038 | lr 0.0005 |Train_Loss 0.5515 |  total_loss_consistency1 0.1002 | Loss 0.6517 | Val_Loss 0.5787| Train Time(s) 0.7292| Val Time(s) 0.0728 | Time(s) 0.8021
2024-12-26 08:18:45,364:INFO::Validation loss decreased (0.578797 --> 0.578656).  Saving model ...
2024-12-26 08:18:46,352:INFO::Epoch_batch_00039 | lr 0.0005 |Train_Loss 0.5503 |  total_loss_consistency1 0.0993 | Loss 0.6496 | Val_Loss 0.5784| Train Time(s) 0.6978| Val Time(s) 0.0473 | Time(s) 0.7451
2024-12-26 08:18:46,352:INFO::Validation loss decreased (0.578656 --> 0.578435).  Saving model ...
2024-12-26 08:18:47,310:INFO::Epoch_batch_00040 | lr 0.0005 |Train_Loss 0.5479 |  total_loss_consistency1 0.0986 | Loss 0.6466 | Val_Loss 0.5782| Train Time(s) 0.6702| Val Time(s) 0.0477 | Time(s) 0.7179
2024-12-26 08:18:47,311:INFO::Validation loss decreased (0.578435 --> 0.578223).  Saving model ...
2024-12-26 08:18:48,549:INFO::Epoch_batch_00041 | lr 0.0005 |Train_Loss 0.5485 |  total_loss_consistency1 0.0977 | Loss 0.6462 | Val_Loss 0.5781| Train Time(s) 0.9122| Val Time(s) 0.0852 | Time(s) 0.9974
2024-12-26 08:18:48,550:INFO::Validation loss decreased (0.578223 --> 0.578123).  Saving model ...
2024-12-26 08:18:49,347:INFO::Epoch_batch_00042 | lr 0.0005 |Train_Loss 0.5473 |  total_loss_consistency1 0.0970 | Loss 0.6443 | Val_Loss 0.5782| Train Time(s) 0.7151| Val Time(s) 0.0792 | Time(s) 0.7943
2024-12-26 08:18:49,347:INFO::EarlyStopping counter: 1 out of 30
2024-12-26 08:18:50,127:INFO::Epoch_batch_00043 | lr 0.0005 |Train_Loss 0.5460 |  total_loss_consistency1 0.0964 | Loss 0.6424 | Val_Loss 0.5782| Train Time(s) 0.7294| Val Time(s) 0.0501 | Time(s) 0.7795
2024-12-26 08:18:50,127:INFO::EarlyStopping counter: 2 out of 30
2024-12-26 08:18:50,910:INFO::Epoch_batch_00044 | lr 0.0005 |Train_Loss 0.5440 |  total_loss_consistency1 0.0957 | Loss 0.6397 | Val_Loss 0.5782| Train Time(s) 0.6965| Val Time(s) 0.0850 | Time(s) 0.7815
2024-12-26 08:18:50,910:INFO::EarlyStopping counter: 3 out of 30
2024-12-26 08:18:51,789:INFO::Epoch_batch_00045 | lr 0.0005 |Train_Loss 0.5436 |  total_loss_consistency1 0.0951 | Loss 0.6387 | Val_Loss 0.5784| Train Time(s) 0.7277| Val Time(s) 0.1488 | Time(s) 0.8765
2024-12-26 08:18:51,789:INFO::EarlyStopping counter: 4 out of 30
2024-12-26 08:18:52,717:INFO::Epoch_batch_00046 | lr 0.0005 |Train_Loss 0.5412 |  total_loss_consistency1 0.0943 | Loss 0.6355 | Val_Loss 0.5785| Train Time(s) 0.8405| Val Time(s) 0.0845 | Time(s) 0.9250
2024-12-26 08:18:52,717:INFO::EarlyStopping counter: 5 out of 30
2024-12-26 08:18:53,494:INFO::Epoch_batch_00047 | lr 0.0005 |Train_Loss 0.5421 |  total_loss_consistency1 0.0936 | Loss 0.6357 | Val_Loss 0.5786| Train Time(s) 0.7264| Val Time(s) 0.0503 | Time(s) 0.7767
2024-12-26 08:18:53,495:INFO::EarlyStopping counter: 6 out of 30
2024-12-26 08:18:54,308:INFO::Epoch_batch_00048 | lr 0.0005 |Train_Loss 0.5406 |  total_loss_consistency1 0.0931 | Loss 0.6337 | Val_Loss 0.5790| Train Time(s) 0.7311| Val Time(s) 0.0817 | Time(s) 0.8128
2024-12-26 08:18:54,308:INFO::EarlyStopping counter: 7 out of 30
2024-12-26 08:18:55,090:INFO::Epoch_batch_00049 | lr 0.0005 |Train_Loss 0.5404 |  total_loss_consistency1 0.0923 | Loss 0.6327 | Val_Loss 0.5794| Train Time(s) 0.6981| Val Time(s) 0.0828 | Time(s) 0.7809
2024-12-26 08:18:55,090:INFO::EarlyStopping counter: 8 out of 30
2024-12-26 08:18:56,138:INFO::Epoch_batch_00050 | lr 0.0005 |Train_Loss 0.5386 |  total_loss_consistency1 0.0916 | Loss 0.6302 | Val_Loss 0.5795| Train Time(s) 0.9324| Val Time(s) 0.1152 | Time(s) 1.0476
2024-12-26 08:18:56,138:INFO::EarlyStopping counter: 9 out of 30
2024-12-26 08:18:56,927:INFO::Epoch_batch_00051 | lr 0.0005 |Train_Loss 0.5386 |  total_loss_consistency1 0.0909 | Loss 0.6296 | Val_Loss 0.5796| Train Time(s) 0.7082| Val Time(s) 0.0795 | Time(s) 0.7877
2024-12-26 08:18:56,927:INFO::EarlyStopping counter: 10 out of 30
2024-12-26 08:18:57,739:INFO::Epoch_batch_00052 | lr 0.0005 |Train_Loss 0.5374 |  total_loss_consistency1 0.0902 | Loss 0.6276 | Val_Loss 0.5796| Train Time(s) 0.7220| Val Time(s) 0.0893 | Time(s) 0.8113
2024-12-26 08:18:57,739:INFO::EarlyStopping counter: 11 out of 30
2024-12-26 08:18:58,551:INFO::Epoch_batch_00053 | lr 0.0005 |Train_Loss 0.5367 |  total_loss_consistency1 0.0893 | Loss 0.6259 | Val_Loss 0.5799| Train Time(s) 0.7284| Val Time(s) 0.0831 | Time(s) 0.8115
2024-12-26 08:18:58,551:INFO::EarlyStopping counter: 12 out of 30
2024-12-26 08:18:59,332:INFO::Epoch_batch_00054 | lr 0.0005 |Train_Loss 0.5354 |  total_loss_consistency1 0.0887 | Loss 0.6241 | Val_Loss 0.5800| Train Time(s) 0.6929| Val Time(s) 0.0873 | Time(s) 0.7802
2024-12-26 08:18:59,332:INFO::EarlyStopping counter: 13 out of 30
2024-12-26 08:19:00,386:INFO::Epoch_batch_00055 | lr 0.0005 |Train_Loss 0.5348 |  total_loss_consistency1 0.0880 | Loss 0.6229 | Val_Loss 0.5793| Train Time(s) 0.9789| Val Time(s) 0.0746 | Time(s) 1.0535
2024-12-26 08:19:00,387:INFO::EarlyStopping counter: 14 out of 30
2024-12-26 08:19:01,203:INFO::Epoch_batch_00056 | lr 0.0005 |Train_Loss 0.5339 |  total_loss_consistency1 0.0873 | Loss 0.6213 | Val_Loss 0.5794| Train Time(s) 0.7436| Val Time(s) 0.0724 | Time(s) 0.8160
2024-12-26 08:19:01,203:INFO::EarlyStopping counter: 15 out of 30
2024-12-26 08:19:02,016:INFO::Epoch_batch_00057 | lr 0.0005 |Train_Loss 0.5337 |  total_loss_consistency1 0.0865 | Loss 0.6202 | Val_Loss 0.5799| Train Time(s) 0.7271| Val Time(s) 0.0848 | Time(s) 0.8119
2024-12-26 08:19:02,016:INFO::EarlyStopping counter: 16 out of 30
2024-12-26 08:19:02,798:INFO::Epoch_batch_00058 | lr 0.0005 |Train_Loss 0.5329 |  total_loss_consistency1 0.0857 | Loss 0.6185 | Val_Loss 0.5807| Train Time(s) 0.6974| Val Time(s) 0.0834 | Time(s) 0.7808
2024-12-26 08:19:02,798:INFO::EarlyStopping counter: 17 out of 30
2024-12-26 08:19:03,676:INFO::Epoch_batch_00059 | lr 0.0005 |Train_Loss 0.5324 |  total_loss_consistency1 0.0849 | Loss 0.6173 | Val_Loss 0.5815| Train Time(s) 0.7275| Val Time(s) 0.1476 | Time(s) 0.8751
2024-12-26 08:19:03,676:INFO::EarlyStopping counter: 18 out of 30
2024-12-26 08:19:04,604:INFO::Epoch_batch_00060 | lr 0.0005 |Train_Loss 0.5313 |  total_loss_consistency1 0.0841 | Loss 0.6154 | Val_Loss 0.5807| Train Time(s) 0.8409| Val Time(s) 0.0848 | Time(s) 0.9257
2024-12-26 08:19:04,605:INFO::EarlyStopping counter: 19 out of 30
2024-12-26 08:19:05,382:INFO::Epoch_batch_00061 | lr 0.0005 |Train_Loss 0.5312 |  total_loss_consistency1 0.0833 | Loss 0.6145 | Val_Loss 0.5797| Train Time(s) 0.7267| Val Time(s) 0.0499 | Time(s) 0.7766
2024-12-26 08:19:05,382:INFO::EarlyStopping counter: 20 out of 30
2024-12-26 08:19:06,193:INFO::Epoch_batch_00062 | lr 0.0005 |Train_Loss 0.5297 |  total_loss_consistency1 0.0824 | Loss 0.6122 | Val_Loss 0.5794| Train Time(s) 0.7272| Val Time(s) 0.0834 | Time(s) 0.8105
2024-12-26 08:19:06,193:INFO::EarlyStopping counter: 21 out of 30
2024-12-26 08:19:06,977:INFO::Epoch_batch_00063 | lr 0.0005 |Train_Loss 0.5299 |  total_loss_consistency1 0.0816 | Loss 0.6115 | Val_Loss 0.5797| Train Time(s) 0.6993| Val Time(s) 0.0834 | Time(s) 0.7827
2024-12-26 08:19:06,977:INFO::EarlyStopping counter: 22 out of 30
2024-12-26 08:19:08,028:INFO::Epoch_batch_00064 | lr 0.0005 |Train_Loss 0.5278 |  total_loss_consistency1 0.0806 | Loss 0.6084 | Val_Loss 0.5807| Train Time(s) 0.9317| Val Time(s) 0.1194 | Time(s) 1.0512
2024-12-26 08:19:08,029:INFO::EarlyStopping counter: 23 out of 30
2024-12-26 08:19:08,818:INFO::Epoch_batch_00065 | lr 0.0005 |Train_Loss 0.5286 |  total_loss_consistency1 0.0794 | Loss 0.6080 | Val_Loss 0.5808| Train Time(s) 0.7139| Val Time(s) 0.0747 | Time(s) 0.7886
2024-12-26 08:19:08,818:INFO::EarlyStopping counter: 24 out of 30
2024-12-26 08:19:09,630:INFO::Epoch_batch_00066 | lr 0.0005 |Train_Loss 0.5290 |  total_loss_consistency1 0.0785 | Loss 0.6075 | Val_Loss 0.5801| Train Time(s) 0.7267| Val Time(s) 0.0851 | Time(s) 0.8118
2024-12-26 08:19:09,631:INFO::EarlyStopping counter: 25 out of 30
2024-12-26 08:19:10,444:INFO::Epoch_batch_00067 | lr 0.0005 |Train_Loss 0.5273 |  total_loss_consistency1 0.0773 | Loss 0.6046 | Val_Loss 0.5799| Train Time(s) 0.7314| Val Time(s) 0.0817 | Time(s) 0.8131
2024-12-26 08:19:10,445:INFO::EarlyStopping counter: 26 out of 30
2024-12-26 08:19:11,224:INFO::Epoch_batch_00068 | lr 0.0005 |Train_Loss 0.5277 |  total_loss_consistency1 0.0761 | Loss 0.6038 | Val_Loss 0.5799| Train Time(s) 0.6915| Val Time(s) 0.0877 | Time(s) 0.7792
2024-12-26 08:19:11,225:INFO::EarlyStopping counter: 27 out of 30
2024-12-26 08:19:12,279:INFO::Epoch_batch_00069 | lr 0.0005 |Train_Loss 0.5265 |  total_loss_consistency1 0.0751 | Loss 0.6016 | Val_Loss 0.5804| Train Time(s) 0.9792| Val Time(s) 0.0748 | Time(s) 1.0540
2024-12-26 08:19:12,279:INFO::EarlyStopping counter: 28 out of 30
2024-12-26 08:19:13,097:INFO::Epoch_batch_00070 | lr 0.0005 |Train_Loss 0.5270 |  total_loss_consistency1 0.0743 | Loss 0.6013 | Val_Loss 0.5803| Train Time(s) 0.7372| Val Time(s) 0.0804 | Time(s) 0.8176
2024-12-26 08:19:13,098:INFO::EarlyStopping counter: 29 out of 30
2024-12-26 08:19:13,912:INFO::Epoch_batch_00071 | lr 0.0005 |Train_Loss 0.5266 |  total_loss_consistency1 0.0728 | Loss 0.5994 | Val_Loss 0.5796| Train Time(s) 0.7314| Val Time(s) 0.0822 | Time(s) 0.8137
2024-12-26 08:19:13,912:INFO::EarlyStopping counter: 30 out of 30
2024-12-26 08:19:13,912:INFO::Eearly stopping!
2024-12-26 08:19:13,995:INFO::
testing...
2024-12-26 08:19:14,223:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:19:14,223:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:19:14,223:INFO::submit dir: submit/submit_simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:19:14,315:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:19:14,329:INFO::{'micro-f1': 0.6911764705882353, 'macro-f1': 0.6579715516966766}
2024-12-26 08:19:14,372:INFO::############### Experiments Stage Ends! ###############
2024-12-26 08:19:14,372:INFO::=============== one experiment stage finish, use 267.6805171966553 time.
2024-12-26 08:19:17,718:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:19:18,766:INFO::Epoch_batch_00000 | lr 0.0005 |Train_Loss 0.7046 |  total_loss_consistency1 0.1771 | Loss 0.8818 | Val_Loss 0.6345| Train Time(s) 0.7180| Val Time(s) 0.0680 | Time(s) 0.7860
2024-12-26 08:19:20,014:INFO::Epoch_batch_00001 | lr 0.0005 |Train_Loss 0.6424 |  total_loss_consistency1 0.1641 | Loss 0.8065 | Val_Loss 0.6318| Train Time(s) 0.9267| Val Time(s) 0.0797 | Time(s) 1.0064
2024-12-26 08:19:20,014:INFO::Validation loss decreased (inf --> 0.631839).  Saving model ...
2024-12-26 08:19:21,059:INFO::Epoch_batch_00002 | lr 0.0005 |Train_Loss 0.6306 |  total_loss_consistency1 0.1569 | Loss 0.7875 | Val_Loss 0.6300| Train Time(s) 0.7126| Val Time(s) 0.0790 | Time(s) 0.7916
2024-12-26 08:19:21,059:INFO::Validation loss decreased (0.631839 --> 0.630018).  Saving model ...
2024-12-26 08:19:22,093:INFO::Epoch_batch_00003 | lr 0.0005 |Train_Loss 0.6308 |  total_loss_consistency1 0.1494 | Loss 0.7802 | Val_Loss 0.6284| Train Time(s) 0.7391| Val Time(s) 0.0499 | Time(s) 0.7890
2024-12-26 08:19:22,093:INFO::Validation loss decreased (0.630018 --> 0.628427).  Saving model ...
2024-12-26 08:19:23,080:INFO::Epoch_batch_00004 | lr 0.0005 |Train_Loss 0.6265 |  total_loss_consistency1 0.1428 | Loss 0.7692 | Val_Loss 0.6267| Train Time(s) 0.6860| Val Time(s) 0.0503 | Time(s) 0.7363
2024-12-26 08:19:23,080:INFO::Validation loss decreased (0.628427 --> 0.626744).  Saving model ...
2024-12-26 08:19:24,336:INFO::Epoch_batch_00005 | lr 0.0005 |Train_Loss 0.6243 |  total_loss_consistency1 0.1374 | Loss 0.7617 | Val_Loss 0.6249| Train Time(s) 0.9655| Val Time(s) 0.0513 | Time(s) 1.0168
2024-12-26 08:19:24,336:INFO::Validation loss decreased (0.626744 --> 0.624888).  Saving model ...
2024-12-26 08:19:25,336:INFO::Epoch_batch_00006 | lr 0.0005 |Train_Loss 0.6235 |  total_loss_consistency1 0.1331 | Loss 0.7567 | Val_Loss 0.6229| Train Time(s) 0.7179| Val Time(s) 0.0499 | Time(s) 0.7677
2024-12-26 08:19:25,337:INFO::Validation loss decreased (0.624888 --> 0.622907).  Saving model ...
2024-12-26 08:19:26,347:INFO::Epoch_batch_00007 | lr 0.0005 |Train_Loss 0.6222 |  total_loss_consistency1 0.1302 | Loss 0.7524 | Val_Loss 0.6208| Train Time(s) 0.7269| Val Time(s) 0.0501 | Time(s) 0.7770
2024-12-26 08:19:26,348:INFO::Validation loss decreased (0.622907 --> 0.620810).  Saving model ...
2024-12-26 08:19:27,359:INFO::Epoch_batch_00008 | lr 0.0005 |Train_Loss 0.6185 |  total_loss_consistency1 0.1281 | Loss 0.7466 | Val_Loss 0.6185| Train Time(s) 0.6968| Val Time(s) 0.0775 | Time(s) 0.7744
2024-12-26 08:19:27,359:INFO::Validation loss decreased (0.620810 --> 0.618515).  Saving model ...
2024-12-26 08:19:28,466:INFO::Epoch_batch_00009 | lr 0.0005 |Train_Loss 0.6165 |  total_loss_consistency1 0.1264 | Loss 0.7428 | Val_Loss 0.6161| Train Time(s) 0.7634| Val Time(s) 0.1068 | Time(s) 0.8701
2024-12-26 08:19:28,466:INFO::Validation loss decreased (0.618515 --> 0.616060).  Saving model ...
2024-12-26 08:19:29,488:INFO::Epoch_batch_00010 | lr 0.0005 |Train_Loss 0.6114 |  total_loss_consistency1 0.1244 | Loss 0.7359 | Val_Loss 0.6134| Train Time(s) 0.7379| Val Time(s) 0.0499 | Time(s) 0.7878
2024-12-26 08:19:29,488:INFO::Validation loss decreased (0.616060 --> 0.613373).  Saving model ...
2024-12-26 08:19:30,504:INFO::Epoch_batch_00011 | lr 0.0005 |Train_Loss 0.6111 |  total_loss_consistency1 0.1228 | Loss 0.7339 | Val_Loss 0.6105| Train Time(s) 0.7289| Val Time(s) 0.0511 | Time(s) 0.7800
2024-12-26 08:19:30,505:INFO::Validation loss decreased (0.613373 --> 0.610542).  Saving model ...
2024-12-26 08:19:31,617:INFO::Epoch_batch_00012 | lr 0.0005 |Train_Loss 0.6089 |  total_loss_consistency1 0.1213 | Loss 0.7303 | Val_Loss 0.6076| Train Time(s) 0.7494| Val Time(s) 0.1166 | Time(s) 0.8660
2024-12-26 08:19:31,617:INFO::Validation loss decreased (0.610542 --> 0.607620).  Saving model ...
2024-12-26 08:19:32,836:INFO::Epoch_batch_00013 | lr 0.0005 |Train_Loss 0.6041 |  total_loss_consistency1 0.1198 | Loss 0.7238 | Val_Loss 0.6045| Train Time(s) 0.9053| Val Time(s) 0.0587 | Time(s) 0.9640
2024-12-26 08:19:32,836:INFO::Validation loss decreased (0.607620 --> 0.604532).  Saving model ...
2024-12-26 08:19:34,062:INFO::Epoch_batch_00014 | lr 0.0005 |Train_Loss 0.6008 |  total_loss_consistency1 0.1184 | Loss 0.7191 | Val_Loss 0.6017| Train Time(s) 0.8540| Val Time(s) 0.1176 | Time(s) 0.9716
2024-12-26 08:19:34,063:INFO::Validation loss decreased (0.604532 --> 0.601656).  Saving model ...
2024-12-26 08:19:35,254:INFO::Epoch_batch_00015 | lr 0.0005 |Train_Loss 0.5966 |  total_loss_consistency1 0.1172 | Loss 0.7138 | Val_Loss 0.5986| Train Time(s) 0.8419| Val Time(s) 0.1056 | Time(s) 0.9475
2024-12-26 08:19:35,254:INFO::Validation loss decreased (0.601656 --> 0.598551).  Saving model ...
2024-12-26 08:19:36,579:INFO::Epoch_batch_00016 | lr 0.0005 |Train_Loss 0.5927 |  total_loss_consistency1 0.1162 | Loss 0.7088 | Val_Loss 0.5953| Train Time(s) 1.0299| Val Time(s) 0.0538 | Time(s) 1.0837
2024-12-26 08:19:36,580:INFO::Validation loss decreased (0.598551 --> 0.595312).  Saving model ...
2024-12-26 08:19:37,963:INFO::Epoch_batch_00017 | lr 0.0005 |Train_Loss 0.5903 |  total_loss_consistency1 0.1151 | Loss 0.7054 | Val_Loss 0.5926| Train Time(s) 1.0924| Val Time(s) 0.0534 | Time(s) 1.1457
2024-12-26 08:19:37,964:INFO::Validation loss decreased (0.595312 --> 0.592554).  Saving model ...
2024-12-26 08:19:39,350:INFO::Epoch_batch_00018 | lr 0.0005 |Train_Loss 0.5870 |  total_loss_consistency1 0.1142 | Loss 0.7013 | Val_Loss 0.5905| Train Time(s) 1.0917| Val Time(s) 0.0540 | Time(s) 1.1457
2024-12-26 08:19:39,350:INFO::Validation loss decreased (0.592554 --> 0.590493).  Saving model ...
2024-12-26 08:19:40,738:INFO::Epoch_batch_00019 | lr 0.0005 |Train_Loss 0.5806 |  total_loss_consistency1 0.1131 | Loss 0.6937 | Val_Loss 0.5886| Train Time(s) 1.0936| Val Time(s) 0.0550 | Time(s) 1.1486
2024-12-26 08:19:40,738:INFO::Validation loss decreased (0.590493 --> 0.588649).  Saving model ...
2024-12-26 08:19:42,112:INFO::Epoch_batch_00020 | lr 0.0005 |Train_Loss 0.5777 |  total_loss_consistency1 0.1125 | Loss 0.6903 | Val_Loss 0.5869| Train Time(s) 1.0184| Val Time(s) 0.1118 | Time(s) 1.1302
2024-12-26 08:19:42,112:INFO::Validation loss decreased (0.588649 --> 0.586860).  Saving model ...
2024-12-26 08:19:43,490:INFO::Epoch_batch_00021 | lr 0.0005 |Train_Loss 0.5745 |  total_loss_consistency1 0.1116 | Loss 0.6862 | Val_Loss 0.5852| Train Time(s) 1.0422| Val Time(s) 0.0956 | Time(s) 1.1378
2024-12-26 08:19:43,490:INFO::Validation loss decreased (0.586860 --> 0.585152).  Saving model ...
2024-12-26 08:19:44,859:INFO::Epoch_batch_00022 | lr 0.0005 |Train_Loss 0.5710 |  total_loss_consistency1 0.1107 | Loss 0.6817 | Val_Loss 0.5839| Train Time(s) 1.0583| Val Time(s) 0.0706 | Time(s) 1.1290
2024-12-26 08:19:44,860:INFO::Validation loss decreased (0.585152 --> 0.583890).  Saving model ...
2024-12-26 08:19:46,236:INFO::Epoch_batch_00023 | lr 0.0005 |Train_Loss 0.5683 |  total_loss_consistency1 0.1101 | Loss 0.6784 | Val_Loss 0.5831| Train Time(s) 1.0815| Val Time(s) 0.0528 | Time(s) 1.1343
2024-12-26 08:19:46,237:INFO::Validation loss decreased (0.583890 --> 0.583116).  Saving model ...
2024-12-26 08:19:47,633:INFO::Epoch_batch_00024 | lr 0.0005 |Train_Loss 0.5662 |  total_loss_consistency1 0.1094 | Loss 0.6756 | Val_Loss 0.5823| Train Time(s) 1.1016| Val Time(s) 0.0536 | Time(s) 1.1551
2024-12-26 08:19:47,633:INFO::Validation loss decreased (0.583116 --> 0.582283).  Saving model ...
2024-12-26 08:19:48,947:INFO::Epoch_batch_00025 | lr 0.0005 |Train_Loss 0.5646 |  total_loss_consistency1 0.1087 | Loss 0.6733 | Val_Loss 0.5816| Train Time(s) 1.0200| Val Time(s) 0.0483 | Time(s) 1.0683
2024-12-26 08:19:48,948:INFO::Validation loss decreased (0.582283 --> 0.581614).  Saving model ...
2024-12-26 08:19:49,971:INFO::Epoch_batch_00026 | lr 0.0005 |Train_Loss 0.5618 |  total_loss_consistency1 0.1076 | Loss 0.6694 | Val_Loss 0.5807| Train Time(s) 0.6783| Val Time(s) 0.0999 | Time(s) 0.7782
2024-12-26 08:19:49,971:INFO::Validation loss decreased (0.581614 --> 0.580685).  Saving model ...
2024-12-26 08:19:50,965:INFO::Epoch_batch_00027 | lr 0.0005 |Train_Loss 0.5589 |  total_loss_consistency1 0.1068 | Loss 0.6657 | Val_Loss 0.5797| Train Time(s) 0.7090| Val Time(s) 0.0503 | Time(s) 0.7593
2024-12-26 08:19:50,966:INFO::Validation loss decreased (0.580685 --> 0.579698).  Saving model ...
2024-12-26 08:19:51,946:INFO::Epoch_batch_00028 | lr 0.0005 |Train_Loss 0.5593 |  total_loss_consistency1 0.1061 | Loss 0.6653 | Val_Loss 0.5786| Train Time(s) 0.6951| Val Time(s) 0.0500 | Time(s) 0.7451
2024-12-26 08:19:51,946:INFO::Validation loss decreased (0.579698 --> 0.578635).  Saving model ...
2024-12-26 08:19:52,928:INFO::Epoch_batch_00029 | lr 0.0005 |Train_Loss 0.5547 |  total_loss_consistency1 0.1052 | Loss 0.6599 | Val_Loss 0.5780| Train Time(s) 0.6950| Val Time(s) 0.0502 | Time(s) 0.7452
2024-12-26 08:19:52,928:INFO::Validation loss decreased (0.578635 --> 0.577987).  Saving model ...
2024-12-26 08:19:53,908:INFO::Epoch_batch_00030 | lr 0.0005 |Train_Loss 0.5545 |  total_loss_consistency1 0.1044 | Loss 0.6590 | Val_Loss 0.5776| Train Time(s) 0.6941| Val Time(s) 0.0504 | Time(s) 0.7445
2024-12-26 08:19:53,908:INFO::Validation loss decreased (0.577987 --> 0.577553).  Saving model ...
2024-12-26 08:19:54,890:INFO::Epoch_batch_00031 | lr 0.0005 |Train_Loss 0.5531 |  total_loss_consistency1 0.1036 | Loss 0.6566 | Val_Loss 0.5771| Train Time(s) 0.6957| Val Time(s) 0.0500 | Time(s) 0.7456
2024-12-26 08:19:54,891:INFO::Validation loss decreased (0.577553 --> 0.577102).  Saving model ...
2024-12-26 08:19:55,874:INFO::Epoch_batch_00032 | lr 0.0005 |Train_Loss 0.5505 |  total_loss_consistency1 0.1030 | Loss 0.6535 | Val_Loss 0.5767| Train Time(s) 0.6979| Val Time(s) 0.0500 | Time(s) 0.7479
2024-12-26 08:19:55,874:INFO::Validation loss decreased (0.577102 --> 0.576728).  Saving model ...
2024-12-26 08:19:56,862:INFO::Epoch_batch_00033 | lr 0.0005 |Train_Loss 0.5508 |  total_loss_consistency1 0.1023 | Loss 0.6531 | Val_Loss 0.5767| Train Time(s) 0.7008| Val Time(s) 0.0503 | Time(s) 0.7511
2024-12-26 08:19:56,863:INFO::Validation loss decreased (0.576728 --> 0.576657).  Saving model ...
2024-12-26 08:19:57,842:INFO::Epoch_batch_00034 | lr 0.0005 |Train_Loss 0.5489 |  total_loss_consistency1 0.1013 | Loss 0.6503 | Val_Loss 0.5764| Train Time(s) 0.6918| Val Time(s) 0.0502 | Time(s) 0.7421
2024-12-26 08:19:57,842:INFO::Validation loss decreased (0.576657 --> 0.576418).  Saving model ...
2024-12-26 08:19:58,825:INFO::Epoch_batch_00035 | lr 0.0005 |Train_Loss 0.5468 |  total_loss_consistency1 0.1006 | Loss 0.6474 | Val_Loss 0.5764| Train Time(s) 0.6927| Val Time(s) 0.0499 | Time(s) 0.7427
2024-12-26 08:19:58,825:INFO::Validation loss decreased (0.576418 --> 0.576409).  Saving model ...
2024-12-26 08:19:59,567:INFO::Epoch_batch_00036 | lr 0.0005 |Train_Loss 0.5446 |  total_loss_consistency1 0.1001 | Loss 0.6447 | Val_Loss 0.5766| Train Time(s) 0.6934| Val Time(s) 0.0480 | Time(s) 0.7413
2024-12-26 08:19:59,567:INFO::EarlyStopping counter: 1 out of 30
2024-12-26 08:20:00,348:INFO::Epoch_batch_00037 | lr 0.0005 |Train_Loss 0.5431 |  total_loss_consistency1 0.0994 | Loss 0.6425 | Val_Loss 0.5768| Train Time(s) 0.6866| Val Time(s) 0.0935 | Time(s) 0.7801
2024-12-26 08:20:00,348:INFO::EarlyStopping counter: 2 out of 30
2024-12-26 08:20:01,131:INFO::Epoch_batch_00038 | lr 0.0005 |Train_Loss 0.5416 |  total_loss_consistency1 0.0986 | Loss 0.6401 | Val_Loss 0.5770| Train Time(s) 0.6983| Val Time(s) 0.0839 | Time(s) 0.7822
2024-12-26 08:20:01,131:INFO::EarlyStopping counter: 3 out of 30
2024-12-26 08:20:01,913:INFO::Epoch_batch_00039 | lr 0.0005 |Train_Loss 0.5411 |  total_loss_consistency1 0.0980 | Loss 0.6391 | Val_Loss 0.5772| Train Time(s) 0.6966| Val Time(s) 0.0846 | Time(s) 0.7812
2024-12-26 08:20:01,913:INFO::EarlyStopping counter: 4 out of 30
2024-12-26 08:20:02,697:INFO::Epoch_batch_00040 | lr 0.0005 |Train_Loss 0.5412 |  total_loss_consistency1 0.0973 | Loss 0.6385 | Val_Loss 0.5770| Train Time(s) 0.6993| Val Time(s) 0.0834 | Time(s) 0.7827
2024-12-26 08:20:02,697:INFO::EarlyStopping counter: 5 out of 30
2024-12-26 08:20:03,479:INFO::Epoch_batch_00041 | lr 0.0005 |Train_Loss 0.5378 |  total_loss_consistency1 0.0969 | Loss 0.6346 | Val_Loss 0.5770| Train Time(s) 0.6975| Val Time(s) 0.0839 | Time(s) 0.7814
2024-12-26 08:20:03,479:INFO::EarlyStopping counter: 6 out of 30
2024-12-26 08:20:04,258:INFO::Epoch_batch_00042 | lr 0.0005 |Train_Loss 0.5381 |  total_loss_consistency1 0.0960 | Loss 0.6342 | Val_Loss 0.5772| Train Time(s) 0.6930| Val Time(s) 0.0856 | Time(s) 0.7786
2024-12-26 08:20:04,259:INFO::EarlyStopping counter: 7 out of 30
2024-12-26 08:20:05,039:INFO::Epoch_batch_00043 | lr 0.0005 |Train_Loss 0.5371 |  total_loss_consistency1 0.0953 | Loss 0.6325 | Val_Loss 0.5776| Train Time(s) 0.6950| Val Time(s) 0.0847 | Time(s) 0.7797
2024-12-26 08:20:05,039:INFO::EarlyStopping counter: 8 out of 30
2024-12-26 08:20:05,820:INFO::Epoch_batch_00044 | lr 0.0005 |Train_Loss 0.5354 |  total_loss_consistency1 0.0947 | Loss 0.6301 | Val_Loss 0.5781| Train Time(s) 0.6957| Val Time(s) 0.0848 | Time(s) 0.7805
2024-12-26 08:20:05,820:INFO::EarlyStopping counter: 9 out of 30
2024-12-26 08:20:06,600:INFO::Epoch_batch_00045 | lr 0.0005 |Train_Loss 0.5346 |  total_loss_consistency1 0.0940 | Loss 0.6286 | Val_Loss 0.5781| Train Time(s) 0.6879| Val Time(s) 0.0912 | Time(s) 0.7791
2024-12-26 08:20:06,600:INFO::EarlyStopping counter: 10 out of 30
2024-12-26 08:20:07,383:INFO::Epoch_batch_00046 | lr 0.0005 |Train_Loss 0.5329 |  total_loss_consistency1 0.0933 | Loss 0.6263 | Val_Loss 0.5780| Train Time(s) 0.6990| Val Time(s) 0.0832 | Time(s) 0.7822
2024-12-26 08:20:07,384:INFO::EarlyStopping counter: 11 out of 30
2024-12-26 08:20:08,167:INFO::Epoch_batch_00047 | lr 0.0005 |Train_Loss 0.5340 |  total_loss_consistency1 0.0927 | Loss 0.6267 | Val_Loss 0.5781| Train Time(s) 0.6997| Val Time(s) 0.0832 | Time(s) 0.7829
2024-12-26 08:20:08,167:INFO::EarlyStopping counter: 12 out of 30
2024-12-26 08:20:08,948:INFO::Epoch_batch_00048 | lr 0.0005 |Train_Loss 0.5326 |  total_loss_consistency1 0.0921 | Loss 0.6246 | Val_Loss 0.5784| Train Time(s) 0.6948| Val Time(s) 0.0853 | Time(s) 0.7801
2024-12-26 08:20:08,948:INFO::EarlyStopping counter: 13 out of 30
2024-12-26 08:20:09,728:INFO::Epoch_batch_00049 | lr 0.0005 |Train_Loss 0.5328 |  total_loss_consistency1 0.0914 | Loss 0.6242 | Val_Loss 0.5788| Train Time(s) 0.6947| Val Time(s) 0.0849 | Time(s) 0.7796
2024-12-26 08:20:09,729:INFO::EarlyStopping counter: 14 out of 30
2024-12-26 08:20:10,511:INFO::Epoch_batch_00050 | lr 0.0005 |Train_Loss 0.5314 |  total_loss_consistency1 0.0906 | Loss 0.6221 | Val_Loss 0.5788| Train Time(s) 0.6988| Val Time(s) 0.0828 | Time(s) 0.7815
2024-12-26 08:20:10,511:INFO::EarlyStopping counter: 15 out of 30
2024-12-26 08:20:11,292:INFO::Epoch_batch_00051 | lr 0.0005 |Train_Loss 0.5305 |  total_loss_consistency1 0.0898 | Loss 0.6203 | Val_Loss 0.5787| Train Time(s) 0.6913| Val Time(s) 0.0892 | Time(s) 0.7804
2024-12-26 08:20:11,293:INFO::EarlyStopping counter: 16 out of 30
2024-12-26 08:20:12,075:INFO::Epoch_batch_00052 | lr 0.0005 |Train_Loss 0.5302 |  total_loss_consistency1 0.0892 | Loss 0.6194 | Val_Loss 0.5785| Train Time(s) 0.6918| Val Time(s) 0.0898 | Time(s) 0.7816
2024-12-26 08:20:12,075:INFO::EarlyStopping counter: 17 out of 30
2024-12-26 08:20:12,854:INFO::Epoch_batch_00053 | lr 0.0005 |Train_Loss 0.5296 |  total_loss_consistency1 0.0887 | Loss 0.6183 | Val_Loss 0.5785| Train Time(s) 0.6850| Val Time(s) 0.0933 | Time(s) 0.7782
2024-12-26 08:20:12,855:INFO::EarlyStopping counter: 18 out of 30
2024-12-26 08:20:13,635:INFO::Epoch_batch_00054 | lr 0.0005 |Train_Loss 0.5285 |  total_loss_consistency1 0.0879 | Loss 0.6164 | Val_Loss 0.5789| Train Time(s) 0.6716| Val Time(s) 0.1078 | Time(s) 0.7794
2024-12-26 08:20:13,635:INFO::EarlyStopping counter: 19 out of 30
2024-12-26 08:20:14,416:INFO::Epoch_batch_00055 | lr 0.0005 |Train_Loss 0.5281 |  total_loss_consistency1 0.0870 | Loss 0.6151 | Val_Loss 0.5791| Train Time(s) 0.6947| Val Time(s) 0.0857 | Time(s) 0.7804
2024-12-26 08:20:14,417:INFO::EarlyStopping counter: 20 out of 30
2024-12-26 08:20:15,199:INFO::Epoch_batch_00056 | lr 0.0005 |Train_Loss 0.5274 |  total_loss_consistency1 0.0862 | Loss 0.6136 | Val_Loss 0.5789| Train Time(s) 0.6985| Val Time(s) 0.0837 | Time(s) 0.7822
2024-12-26 08:20:15,200:INFO::EarlyStopping counter: 21 out of 30
2024-12-26 08:20:15,982:INFO::Epoch_batch_00057 | lr 0.0005 |Train_Loss 0.5268 |  total_loss_consistency1 0.0854 | Loss 0.6122 | Val_Loss 0.5788| Train Time(s) 0.6988| Val Time(s) 0.0835 | Time(s) 0.7823
2024-12-26 08:20:15,983:INFO::EarlyStopping counter: 22 out of 30
2024-12-26 08:20:16,765:INFO::Epoch_batch_00058 | lr 0.0005 |Train_Loss 0.5260 |  total_loss_consistency1 0.0846 | Loss 0.6106 | Val_Loss 0.5789| Train Time(s) 0.6927| Val Time(s) 0.0882 | Time(s) 0.7808
2024-12-26 08:20:16,765:INFO::EarlyStopping counter: 23 out of 30
2024-12-26 08:20:17,546:INFO::Epoch_batch_00059 | lr 0.0005 |Train_Loss 0.5260 |  total_loss_consistency1 0.0838 | Loss 0.6097 | Val_Loss 0.5794| Train Time(s) 0.6935| Val Time(s) 0.0869 | Time(s) 0.7804
2024-12-26 08:20:17,546:INFO::EarlyStopping counter: 24 out of 30
2024-12-26 08:20:18,329:INFO::Epoch_batch_00060 | lr 0.0005 |Train_Loss 0.5250 |  total_loss_consistency1 0.0829 | Loss 0.6079 | Val_Loss 0.5797| Train Time(s) 0.6966| Val Time(s) 0.0856 | Time(s) 0.7823
2024-12-26 08:20:18,330:INFO::EarlyStopping counter: 25 out of 30
2024-12-26 08:20:19,111:INFO::Epoch_batch_00061 | lr 0.0005 |Train_Loss 0.5251 |  total_loss_consistency1 0.0819 | Loss 0.6070 | Val_Loss 0.5797| Train Time(s) 0.6972| Val Time(s) 0.0835 | Time(s) 0.7807
2024-12-26 08:20:19,111:INFO::EarlyStopping counter: 26 out of 30
2024-12-26 08:20:19,892:INFO::Epoch_batch_00062 | lr 0.0005 |Train_Loss 0.5247 |  total_loss_consistency1 0.0807 | Loss 0.6054 | Val_Loss 0.5798| Train Time(s) 0.6954| Val Time(s) 0.0849 | Time(s) 0.7804
2024-12-26 08:20:19,892:INFO::EarlyStopping counter: 27 out of 30
2024-12-26 08:20:20,676:INFO::Epoch_batch_00063 | lr 0.0005 |Train_Loss 0.5247 |  total_loss_consistency1 0.0800 | Loss 0.6047 | Val_Loss 0.5801| Train Time(s) 0.6999| Val Time(s) 0.0833 | Time(s) 0.7832
2024-12-26 08:20:20,676:INFO::EarlyStopping counter: 28 out of 30
2024-12-26 08:20:21,458:INFO::Epoch_batch_00064 | lr 0.0005 |Train_Loss 0.5243 |  total_loss_consistency1 0.0790 | Loss 0.6034 | Val_Loss 0.5805| Train Time(s) 0.6987| Val Time(s) 0.0824 | Time(s) 0.7811
2024-12-26 08:20:21,458:INFO::EarlyStopping counter: 29 out of 30
2024-12-26 08:20:22,239:INFO::Epoch_batch_00065 | lr 0.0005 |Train_Loss 0.5239 |  total_loss_consistency1 0.0778 | Loss 0.6017 | Val_Loss 0.5810| Train Time(s) 0.6878| Val Time(s) 0.0918 | Time(s) 0.7796
2024-12-26 08:20:22,239:INFO::EarlyStopping counter: 30 out of 30
2024-12-26 08:20:22,239:INFO::Eearly stopping!
2024-12-26 08:20:22,329:INFO::
testing...
2024-12-26 08:20:22,558:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:20:22,558:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:20:22,558:INFO::submit dir: submit/submit_simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:20:22,664:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:20:22,678:INFO::{'micro-f1': 0.6937431394072447, 'macro-f1': 0.6616467869967048}
2024-12-26 08:20:22,725:INFO::############### Experiments Stage Ends! ###############
2024-12-26 08:20:22,725:INFO::=============== one experiment stage finish, use 336.0338463783264 time.
2024-12-26 08:20:26,069:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:20:27,116:INFO::Epoch_batch_00000 | lr 0.0005 |Train_Loss 0.7472 |  total_loss_consistency1 0.1754 | Loss 0.9225 | Val_Loss 0.6423| Train Time(s) 0.7348| Val Time(s) 0.0499 | Time(s) 0.7848
2024-12-26 08:20:28,090:INFO::Epoch_batch_00001 | lr 0.0005 |Train_Loss 0.6446 |  total_loss_consistency1 0.1685 | Loss 0.8131 | Val_Loss 0.6382| Train Time(s) 0.6858| Val Time(s) 0.0496 | Time(s) 0.7355
2024-12-26 08:20:28,090:INFO::Validation loss decreased (inf --> 0.638161).  Saving model ...
2024-12-26 08:20:29,066:INFO::Epoch_batch_00002 | lr 0.0005 |Train_Loss 0.6352 |  total_loss_consistency1 0.1608 | Loss 0.7960 | Val_Loss 0.6358| Train Time(s) 0.6878| Val Time(s) 0.0496 | Time(s) 0.7374
2024-12-26 08:20:29,066:INFO::Validation loss decreased (0.638161 --> 0.635761).  Saving model ...
2024-12-26 08:20:30,048:INFO::Epoch_batch_00003 | lr 0.0005 |Train_Loss 0.6325 |  total_loss_consistency1 0.1543 | Loss 0.7869 | Val_Loss 0.6340| Train Time(s) 0.6942| Val Time(s) 0.0502 | Time(s) 0.7443
2024-12-26 08:20:30,048:INFO::Validation loss decreased (0.635761 --> 0.634035).  Saving model ...
2024-12-26 08:20:31,028:INFO::Epoch_batch_00004 | lr 0.0005 |Train_Loss 0.6288 |  total_loss_consistency1 0.1489 | Loss 0.7777 | Val_Loss 0.6327| Train Time(s) 0.6924| Val Time(s) 0.0494 | Time(s) 0.7418
2024-12-26 08:20:31,028:INFO::Validation loss decreased (0.634035 --> 0.632664).  Saving model ...
2024-12-26 08:20:32,010:INFO::Epoch_batch_00005 | lr 0.0005 |Train_Loss 0.6278 |  total_loss_consistency1 0.1440 | Loss 0.7718 | Val_Loss 0.6315| Train Time(s) 0.6934| Val Time(s) 0.0499 | Time(s) 0.7433
2024-12-26 08:20:32,011:INFO::Validation loss decreased (0.632664 --> 0.631503).  Saving model ...
2024-12-26 08:20:32,985:INFO::Epoch_batch_00006 | lr 0.0005 |Train_Loss 0.6273 |  total_loss_consistency1 0.1394 | Loss 0.7666 | Val_Loss 0.6304| Train Time(s) 0.6878| Val Time(s) 0.0495 | Time(s) 0.7372
2024-12-26 08:20:32,985:INFO::Validation loss decreased (0.631503 --> 0.630374).  Saving model ...
2024-12-26 08:20:33,964:INFO::Epoch_batch_00007 | lr 0.0005 |Train_Loss 0.6266 |  total_loss_consistency1 0.1355 | Loss 0.7621 | Val_Loss 0.6292| Train Time(s) 0.6908| Val Time(s) 0.0494 | Time(s) 0.7402
2024-12-26 08:20:33,964:INFO::Validation loss decreased (0.630374 --> 0.629191).  Saving model ...
2024-12-26 08:20:34,941:INFO::Epoch_batch_00008 | lr 0.0005 |Train_Loss 0.6260 |  total_loss_consistency1 0.1323 | Loss 0.7583 | Val_Loss 0.6280| Train Time(s) 0.6910| Val Time(s) 0.0495 | Time(s) 0.7405
2024-12-26 08:20:34,942:INFO::Validation loss decreased (0.629191 --> 0.627982).  Saving model ...
2024-12-26 08:20:35,920:INFO::Epoch_batch_00009 | lr 0.0005 |Train_Loss 0.6236 |  total_loss_consistency1 0.1297 | Loss 0.7534 | Val_Loss 0.6267| Train Time(s) 0.6926| Val Time(s) 0.0497 | Time(s) 0.7422
2024-12-26 08:20:35,920:INFO::Validation loss decreased (0.627982 --> 0.626701).  Saving model ...
2024-12-26 08:20:36,899:INFO::Epoch_batch_00010 | lr 0.0005 |Train_Loss 0.6211 |  total_loss_consistency1 0.1272 | Loss 0.7483 | Val_Loss 0.6254| Train Time(s) 0.6956| Val Time(s) 0.0469 | Time(s) 0.7425
2024-12-26 08:20:36,900:INFO::Validation loss decreased (0.626701 --> 0.625352).  Saving model ...
2024-12-26 08:20:37,870:INFO::Epoch_batch_00011 | lr 0.0005 |Train_Loss 0.6195 |  total_loss_consistency1 0.1248 | Loss 0.7443 | Val_Loss 0.6239| Train Time(s) 0.6857| Val Time(s) 0.0493 | Time(s) 0.7350
2024-12-26 08:20:37,871:INFO::Validation loss decreased (0.625352 --> 0.623877).  Saving model ...
2024-12-26 08:20:38,850:INFO::Epoch_batch_00012 | lr 0.0005 |Train_Loss 0.6195 |  total_loss_consistency1 0.1224 | Loss 0.7419 | Val_Loss 0.6223| Train Time(s) 0.6920| Val Time(s) 0.0503 | Time(s) 0.7423
2024-12-26 08:20:38,851:INFO::Validation loss decreased (0.623877 --> 0.622281).  Saving model ...
2024-12-26 08:20:39,829:INFO::Epoch_batch_00013 | lr 0.0005 |Train_Loss 0.6166 |  total_loss_consistency1 0.1206 | Loss 0.7372 | Val_Loss 0.6205| Train Time(s) 0.6933| Val Time(s) 0.0496 | Time(s) 0.7429
2024-12-26 08:20:39,829:INFO::Validation loss decreased (0.622281 --> 0.620499).  Saving model ...
2024-12-26 08:20:40,813:INFO::Epoch_batch_00014 | lr 0.0005 |Train_Loss 0.6151 |  total_loss_consistency1 0.1189 | Loss 0.7340 | Val_Loss 0.6185| Train Time(s) 0.6941| Val Time(s) 0.0495 | Time(s) 0.7435
2024-12-26 08:20:40,813:INFO::Validation loss decreased (0.620499 --> 0.618451).  Saving model ...
2024-12-26 08:20:41,794:INFO::Epoch_batch_00015 | lr 0.0005 |Train_Loss 0.6125 |  total_loss_consistency1 0.1175 | Loss 0.7299 | Val_Loss 0.6161| Train Time(s) 0.6930| Val Time(s) 0.0497 | Time(s) 0.7427
2024-12-26 08:20:41,794:INFO::Validation loss decreased (0.618451 --> 0.616083).  Saving model ...
2024-12-26 08:20:42,771:INFO::Epoch_batch_00016 | lr 0.0005 |Train_Loss 0.6100 |  total_loss_consistency1 0.1166 | Loss 0.7266 | Val_Loss 0.6133| Train Time(s) 0.6899| Val Time(s) 0.0497 | Time(s) 0.7395
2024-12-26 08:20:42,771:INFO::Validation loss decreased (0.616083 --> 0.613345).  Saving model ...
2024-12-26 08:20:43,750:INFO::Epoch_batch_00017 | lr 0.0005 |Train_Loss 0.6068 |  total_loss_consistency1 0.1157 | Loss 0.7225 | Val_Loss 0.6103| Train Time(s) 0.6914| Val Time(s) 0.0500 | Time(s) 0.7414
2024-12-26 08:20:43,751:INFO::Validation loss decreased (0.613345 --> 0.610281).  Saving model ...
2024-12-26 08:20:44,728:INFO::Epoch_batch_00018 | lr 0.0005 |Train_Loss 0.6034 |  total_loss_consistency1 0.1148 | Loss 0.7182 | Val_Loss 0.6068| Train Time(s) 0.6906| Val Time(s) 0.0498 | Time(s) 0.7404
2024-12-26 08:20:44,728:INFO::Validation loss decreased (0.610281 --> 0.606824).  Saving model ...
2024-12-26 08:20:45,710:INFO::Epoch_batch_00019 | lr 0.0005 |Train_Loss 0.6003 |  total_loss_consistency1 0.1141 | Loss 0.7144 | Val_Loss 0.6030| Train Time(s) 0.6948| Val Time(s) 0.0499 | Time(s) 0.7446
2024-12-26 08:20:45,710:INFO::Validation loss decreased (0.606824 --> 0.603038).  Saving model ...
2024-12-26 08:20:46,691:INFO::Epoch_batch_00020 | lr 0.0005 |Train_Loss 0.5957 |  total_loss_consistency1 0.1135 | Loss 0.7092 | Val_Loss 0.5991| Train Time(s) 0.6924| Val Time(s) 0.0499 | Time(s) 0.7424
2024-12-26 08:20:46,691:INFO::Validation loss decreased (0.603038 --> 0.599061).  Saving model ...
2024-12-26 08:20:47,672:INFO::Epoch_batch_00021 | lr 0.0005 |Train_Loss 0.5924 |  total_loss_consistency1 0.1128 | Loss 0.7051 | Val_Loss 0.5950| Train Time(s) 0.6945| Val Time(s) 0.0500 | Time(s) 0.7445
2024-12-26 08:20:47,672:INFO::Validation loss decreased (0.599061 --> 0.594991).  Saving model ...
2024-12-26 08:20:48,647:INFO::Epoch_batch_00022 | lr 0.0005 |Train_Loss 0.5880 |  total_loss_consistency1 0.1123 | Loss 0.7003 | Val_Loss 0.5912| Train Time(s) 0.6897| Val Time(s) 0.0498 | Time(s) 0.7395
2024-12-26 08:20:48,648:INFO::Validation loss decreased (0.594991 --> 0.591237).  Saving model ...
2024-12-26 08:20:49,628:INFO::Epoch_batch_00023 | lr 0.0005 |Train_Loss 0.5818 |  total_loss_consistency1 0.1120 | Loss 0.6938 | Val_Loss 0.5884| Train Time(s) 0.6923| Val Time(s) 0.0497 | Time(s) 0.7420
2024-12-26 08:20:49,628:INFO::Validation loss decreased (0.591237 --> 0.588366).  Saving model ...
2024-12-26 08:20:50,608:INFO::Epoch_batch_00024 | lr 0.0005 |Train_Loss 0.5791 |  total_loss_consistency1 0.1112 | Loss 0.6903 | Val_Loss 0.5870| Train Time(s) 0.6930| Val Time(s) 0.0490 | Time(s) 0.7419
2024-12-26 08:20:50,608:INFO::Validation loss decreased (0.588366 --> 0.586951).  Saving model ...
2024-12-26 08:20:51,591:INFO::Epoch_batch_00025 | lr 0.0005 |Train_Loss 0.5760 |  total_loss_consistency1 0.1104 | Loss 0.6864 | Val_Loss 0.5859| Train Time(s) 0.6952| Val Time(s) 0.0498 | Time(s) 0.7450
2024-12-26 08:20:51,591:INFO::Validation loss decreased (0.586951 --> 0.585935).  Saving model ...
2024-12-26 08:20:52,573:INFO::Epoch_batch_00026 | lr 0.0005 |Train_Loss 0.5731 |  total_loss_consistency1 0.1093 | Loss 0.6824 | Val_Loss 0.5846| Train Time(s) 0.6948| Val Time(s) 0.0493 | Time(s) 0.7442
2024-12-26 08:20:52,573:INFO::Validation loss decreased (0.585935 --> 0.584624).  Saving model ...
2024-12-26 08:20:53,551:INFO::Epoch_batch_00027 | lr 0.0005 |Train_Loss 0.5715 |  total_loss_consistency1 0.1085 | Loss 0.6800 | Val_Loss 0.5826| Train Time(s) 0.6917| Val Time(s) 0.0495 | Time(s) 0.7413
2024-12-26 08:20:53,552:INFO::Validation loss decreased (0.584624 --> 0.582621).  Saving model ...
2024-12-26 08:20:54,532:INFO::Epoch_batch_00028 | lr 0.0005 |Train_Loss 0.5673 |  total_loss_consistency1 0.1074 | Loss 0.6747 | Val_Loss 0.5808| Train Time(s) 0.6929| Val Time(s) 0.0497 | Time(s) 0.7427
2024-12-26 08:20:54,532:INFO::Validation loss decreased (0.582621 --> 0.580846).  Saving model ...
2024-12-26 08:20:55,509:INFO::Epoch_batch_00029 | lr 0.0005 |Train_Loss 0.5648 |  total_loss_consistency1 0.1067 | Loss 0.6715 | Val_Loss 0.5799| Train Time(s) 0.6917| Val Time(s) 0.0501 | Time(s) 0.7418
2024-12-26 08:20:55,510:INFO::Validation loss decreased (0.580846 --> 0.579924).  Saving model ...
2024-12-26 08:20:56,489:INFO::Epoch_batch_00030 | lr 0.0005 |Train_Loss 0.5613 |  total_loss_consistency1 0.1062 | Loss 0.6674 | Val_Loss 0.5792| Train Time(s) 0.6935| Val Time(s) 0.0498 | Time(s) 0.7433
2024-12-26 08:20:56,489:INFO::Validation loss decreased (0.579924 --> 0.579239).  Saving model ...
2024-12-26 08:20:57,467:INFO::Epoch_batch_00031 | lr 0.0005 |Train_Loss 0.5602 |  total_loss_consistency1 0.1051 | Loss 0.6653 | Val_Loss 0.5789| Train Time(s) 0.6931| Val Time(s) 0.0498 | Time(s) 0.7429
2024-12-26 08:20:57,468:INFO::Validation loss decreased (0.579239 --> 0.578857).  Saving model ...
2024-12-26 08:20:58,447:INFO::Epoch_batch_00032 | lr 0.0005 |Train_Loss 0.5596 |  total_loss_consistency1 0.1045 | Loss 0.6641 | Val_Loss 0.5786| Train Time(s) 0.6922| Val Time(s) 0.0499 | Time(s) 0.7421
2024-12-26 08:20:58,447:INFO::Validation loss decreased (0.578857 --> 0.578643).  Saving model ...
2024-12-26 08:20:59,428:INFO::Epoch_batch_00033 | lr 0.0005 |Train_Loss 0.5580 |  total_loss_consistency1 0.1035 | Loss 0.6615 | Val_Loss 0.5786| Train Time(s) 0.6940| Val Time(s) 0.0501 | Time(s) 0.7440
2024-12-26 08:20:59,429:INFO::Validation loss decreased (0.578643 --> 0.578575).  Saving model ...
2024-12-26 08:21:00,174:INFO::Epoch_batch_00034 | lr 0.0005 |Train_Loss 0.5553 |  total_loss_consistency1 0.1025 | Loss 0.6578 | Val_Loss 0.5787| Train Time(s) 0.6949| Val Time(s) 0.0500 | Time(s) 0.7449
2024-12-26 08:21:00,174:INFO::EarlyStopping counter: 1 out of 30
2024-12-26 08:21:01,192:INFO::Epoch_batch_00035 | lr 0.0005 |Train_Loss 0.5525 |  total_loss_consistency1 0.1015 | Loss 0.6540 | Val_Loss 0.5784| Train Time(s) 0.6983| Val Time(s) 0.0827 | Time(s) 0.7811
2024-12-26 08:21:01,192:INFO::Validation loss decreased (0.578575 --> 0.578419).  Saving model ...
2024-12-26 08:21:02,169:INFO::Epoch_batch_00036 | lr 0.0005 |Train_Loss 0.5533 |  total_loss_consistency1 0.1007 | Loss 0.6539 | Val_Loss 0.5781| Train Time(s) 0.6919| Val Time(s) 0.0495 | Time(s) 0.7415
2024-12-26 08:21:02,169:INFO::Validation loss decreased (0.578419 --> 0.578133).  Saving model ...
2024-12-26 08:21:03,148:INFO::Epoch_batch_00037 | lr 0.0005 |Train_Loss 0.5515 |  total_loss_consistency1 0.0996 | Loss 0.6511 | Val_Loss 0.5780| Train Time(s) 0.6919| Val Time(s) 0.0499 | Time(s) 0.7417
2024-12-26 08:21:03,148:INFO::Validation loss decreased (0.578133 --> 0.577981).  Saving model ...
2024-12-26 08:21:04,123:INFO::Epoch_batch_00038 | lr 0.0005 |Train_Loss 0.5507 |  total_loss_consistency1 0.0989 | Loss 0.6496 | Val_Loss 0.5778| Train Time(s) 0.6894| Val Time(s) 0.0493 | Time(s) 0.7387
2024-12-26 08:21:04,123:INFO::Validation loss decreased (0.577981 --> 0.577849).  Saving model ...
2024-12-26 08:21:04,866:INFO::Epoch_batch_00039 | lr 0.0005 |Train_Loss 0.5478 |  total_loss_consistency1 0.0980 | Loss 0.6458 | Val_Loss 0.5779| Train Time(s) 0.6921| Val Time(s) 0.0500 | Time(s) 0.7422
2024-12-26 08:21:04,866:INFO::EarlyStopping counter: 1 out of 30
2024-12-26 08:21:05,648:INFO::Epoch_batch_00040 | lr 0.0005 |Train_Loss 0.5468 |  total_loss_consistency1 0.0972 | Loss 0.6440 | Val_Loss 0.5781| Train Time(s) 0.7000| Val Time(s) 0.0814 | Time(s) 0.7814
2024-12-26 08:21:05,649:INFO::EarlyStopping counter: 2 out of 30
2024-12-26 08:21:06,428:INFO::Epoch_batch_00041 | lr 0.0005 |Train_Loss 0.5463 |  total_loss_consistency1 0.0961 | Loss 0.6424 | Val_Loss 0.5783| Train Time(s) 0.6874| Val Time(s) 0.0916 | Time(s) 0.7791
2024-12-26 08:21:06,429:INFO::EarlyStopping counter: 3 out of 30
2024-12-26 08:21:07,210:INFO::Epoch_batch_00042 | lr 0.0005 |Train_Loss 0.5446 |  total_loss_consistency1 0.0954 | Loss 0.6400 | Val_Loss 0.5783| Train Time(s) 0.6974| Val Time(s) 0.0833 | Time(s) 0.7807
2024-12-26 08:21:07,210:INFO::EarlyStopping counter: 4 out of 30
2024-12-26 08:21:07,991:INFO::Epoch_batch_00043 | lr 0.0005 |Train_Loss 0.5421 |  total_loss_consistency1 0.0947 | Loss 0.6368 | Val_Loss 0.5783| Train Time(s) 0.6959| Val Time(s) 0.0845 | Time(s) 0.7805
2024-12-26 08:21:07,992:INFO::EarlyStopping counter: 5 out of 30
2024-12-26 08:21:08,772:INFO::Epoch_batch_00044 | lr 0.0005 |Train_Loss 0.5429 |  total_loss_consistency1 0.0940 | Loss 0.6369 | Val_Loss 0.5784| Train Time(s) 0.6947| Val Time(s) 0.0847 | Time(s) 0.7794
2024-12-26 08:21:08,772:INFO::EarlyStopping counter: 6 out of 30
2024-12-26 08:21:09,552:INFO::Epoch_batch_00045 | lr 0.0005 |Train_Loss 0.5405 |  total_loss_consistency1 0.0930 | Loss 0.6335 | Val_Loss 0.5783| Train Time(s) 0.6950| Val Time(s) 0.0849 | Time(s) 0.7799
2024-12-26 08:21:09,553:INFO::EarlyStopping counter: 7 out of 30
2024-12-26 08:21:10,334:INFO::Epoch_batch_00046 | lr 0.0005 |Train_Loss 0.5404 |  total_loss_consistency1 0.0925 | Loss 0.6329 | Val_Loss 0.5782| Train Time(s) 0.6930| Val Time(s) 0.0875 | Time(s) 0.7806
2024-12-26 08:21:10,334:INFO::EarlyStopping counter: 8 out of 30
2024-12-26 08:21:11,116:INFO::Epoch_batch_00047 | lr 0.0005 |Train_Loss 0.5390 |  total_loss_consistency1 0.0918 | Loss 0.6308 | Val_Loss 0.5783| Train Time(s) 0.6959| Val Time(s) 0.0857 | Time(s) 0.7816
2024-12-26 08:21:11,116:INFO::EarlyStopping counter: 9 out of 30
2024-12-26 08:21:11,897:INFO::Epoch_batch_00048 | lr 0.0005 |Train_Loss 0.5373 |  total_loss_consistency1 0.0910 | Loss 0.6283 | Val_Loss 0.5781| Train Time(s) 0.6950| Val Time(s) 0.0853 | Time(s) 0.7803
2024-12-26 08:21:11,897:INFO::EarlyStopping counter: 10 out of 30
2024-12-26 08:21:12,678:INFO::Epoch_batch_00049 | lr 0.0005 |Train_Loss 0.5380 |  total_loss_consistency1 0.0902 | Loss 0.6282 | Val_Loss 0.5783| Train Time(s) 0.6955| Val Time(s) 0.0847 | Time(s) 0.7801
2024-12-26 08:21:12,678:INFO::EarlyStopping counter: 11 out of 30
2024-12-26 08:21:13,459:INFO::Epoch_batch_00050 | lr 0.0005 |Train_Loss 0.5362 |  total_loss_consistency1 0.0894 | Loss 0.6257 | Val_Loss 0.5787| Train Time(s) 0.6948| Val Time(s) 0.0849 | Time(s) 0.7797
2024-12-26 08:21:13,459:INFO::EarlyStopping counter: 12 out of 30
2024-12-26 08:21:14,241:INFO::Epoch_batch_00051 | lr 0.0005 |Train_Loss 0.5366 |  total_loss_consistency1 0.0888 | Loss 0.6254 | Val_Loss 0.5793| Train Time(s) 0.6980| Val Time(s) 0.0836 | Time(s) 0.7816
2024-12-26 08:21:14,241:INFO::EarlyStopping counter: 13 out of 30
2024-12-26 08:21:15,022:INFO::Epoch_batch_00052 | lr 0.0005 |Train_Loss 0.5341 |  total_loss_consistency1 0.0878 | Loss 0.6220 | Val_Loss 0.5796| Train Time(s) 0.6950| Val Time(s) 0.0852 | Time(s) 0.7802
2024-12-26 08:21:15,022:INFO::EarlyStopping counter: 14 out of 30
2024-12-26 08:21:15,802:INFO::Epoch_batch_00053 | lr 0.0005 |Train_Loss 0.5350 |  total_loss_consistency1 0.0871 | Loss 0.6221 | Val_Loss 0.5793| Train Time(s) 0.6935| Val Time(s) 0.0859 | Time(s) 0.7793
2024-12-26 08:21:15,802:INFO::EarlyStopping counter: 15 out of 30
2024-12-26 08:21:16,585:INFO::Epoch_batch_00054 | lr 0.0005 |Train_Loss 0.5333 |  total_loss_consistency1 0.0862 | Loss 0.6194 | Val_Loss 0.5789| Train Time(s) 0.7006| Val Time(s) 0.0813 | Time(s) 0.7819
2024-12-26 08:21:16,585:INFO::EarlyStopping counter: 16 out of 30
2024-12-26 08:21:17,367:INFO::Epoch_batch_00055 | lr 0.0005 |Train_Loss 0.5324 |  total_loss_consistency1 0.0854 | Loss 0.6178 | Val_Loss 0.5789| Train Time(s) 0.6959| Val Time(s) 0.0849 | Time(s) 0.7808
2024-12-26 08:21:17,367:INFO::EarlyStopping counter: 17 out of 30
2024-12-26 08:21:18,147:INFO::Epoch_batch_00056 | lr 0.0005 |Train_Loss 0.5319 |  total_loss_consistency1 0.0846 | Loss 0.6165 | Val_Loss 0.5792| Train Time(s) 0.6934| Val Time(s) 0.0859 | Time(s) 0.7793
2024-12-26 08:21:18,147:INFO::EarlyStopping counter: 18 out of 30
2024-12-26 08:21:18,930:INFO::Epoch_batch_00057 | lr 0.0005 |Train_Loss 0.5303 |  total_loss_consistency1 0.0836 | Loss 0.6139 | Val_Loss 0.5794| Train Time(s) 0.6982| Val Time(s) 0.0840 | Time(s) 0.7822
2024-12-26 08:21:18,930:INFO::EarlyStopping counter: 19 out of 30
2024-12-26 08:21:19,714:INFO::Epoch_batch_00058 | lr 0.0005 |Train_Loss 0.5310 |  total_loss_consistency1 0.0826 | Loss 0.6136 | Val_Loss 0.5791| Train Time(s) 0.7000| Val Time(s) 0.0830 | Time(s) 0.7830
2024-12-26 08:21:19,714:INFO::EarlyStopping counter: 20 out of 30
2024-12-26 08:21:20,494:INFO::Epoch_batch_00059 | lr 0.0005 |Train_Loss 0.5296 |  total_loss_consistency1 0.0816 | Loss 0.6111 | Val_Loss 0.5792| Train Time(s) 0.6908| Val Time(s) 0.0884 | Time(s) 0.7792
2024-12-26 08:21:20,494:INFO::EarlyStopping counter: 21 out of 30
2024-12-26 08:21:21,274:INFO::Epoch_batch_00060 | lr 0.0005 |Train_Loss 0.5291 |  total_loss_consistency1 0.0805 | Loss 0.6097 | Val_Loss 0.5793| Train Time(s) 0.6921| Val Time(s) 0.0877 | Time(s) 0.7798
2024-12-26 08:21:21,275:INFO::EarlyStopping counter: 22 out of 30
2024-12-26 08:21:22,056:INFO::Epoch_batch_00061 | lr 0.0005 |Train_Loss 0.5290 |  total_loss_consistency1 0.0797 | Loss 0.6087 | Val_Loss 0.5792| Train Time(s) 0.6935| Val Time(s) 0.0870 | Time(s) 0.7805
2024-12-26 08:21:22,056:INFO::EarlyStopping counter: 23 out of 30
2024-12-26 08:21:22,837:INFO::Epoch_batch_00062 | lr 0.0005 |Train_Loss 0.5284 |  total_loss_consistency1 0.0785 | Loss 0.6069 | Val_Loss 0.5794| Train Time(s) 0.6975| Val Time(s) 0.0830 | Time(s) 0.7805
2024-12-26 08:21:22,837:INFO::EarlyStopping counter: 24 out of 30
2024-12-26 08:21:23,618:INFO::Epoch_batch_00063 | lr 0.0005 |Train_Loss 0.5280 |  total_loss_consistency1 0.0773 | Loss 0.6053 | Val_Loss 0.5793| Train Time(s) 0.6955| Val Time(s) 0.0848 | Time(s) 0.7802
2024-12-26 08:21:23,618:INFO::EarlyStopping counter: 25 out of 30
2024-12-26 08:21:24,400:INFO::Epoch_batch_00064 | lr 0.0005 |Train_Loss 0.5270 |  total_loss_consistency1 0.0761 | Loss 0.6031 | Val_Loss 0.5791| Train Time(s) 0.6969| Val Time(s) 0.0837 | Time(s) 0.7807
2024-12-26 08:21:24,400:INFO::EarlyStopping counter: 26 out of 30
2024-12-26 08:21:25,179:INFO::Epoch_batch_00065 | lr 0.0005 |Train_Loss 0.5259 |  total_loss_consistency1 0.0747 | Loss 0.6007 | Val_Loss 0.5790| Train Time(s) 0.6914| Val Time(s) 0.0876 | Time(s) 0.7789
2024-12-26 08:21:25,180:INFO::EarlyStopping counter: 27 out of 30
2024-12-26 08:21:25,962:INFO::Epoch_batch_00066 | lr 0.0005 |Train_Loss 0.5277 |  total_loss_consistency1 0.0735 | Loss 0.6012 | Val_Loss 0.5789| Train Time(s) 0.6990| Val Time(s) 0.0830 | Time(s) 0.7820
2024-12-26 08:21:25,963:INFO::EarlyStopping counter: 28 out of 30
2024-12-26 08:21:26,746:INFO::Epoch_batch_00067 | lr 0.0005 |Train_Loss 0.5263 |  total_loss_consistency1 0.0719 | Loss 0.5982 | Val_Loss 0.5791| Train Time(s) 0.7023| Val Time(s) 0.0809 | Time(s) 0.7833
2024-12-26 08:21:26,747:INFO::EarlyStopping counter: 29 out of 30
2024-12-26 08:21:27,527:INFO::Epoch_batch_00068 | lr 0.0005 |Train_Loss 0.5253 |  total_loss_consistency1 0.0706 | Loss 0.5960 | Val_Loss 0.5796| Train Time(s) 0.6954| Val Time(s) 0.0846 | Time(s) 0.7800
2024-12-26 08:21:27,527:INFO::EarlyStopping counter: 30 out of 30
2024-12-26 08:21:27,528:INFO::Eearly stopping!
2024-12-26 08:21:27,612:INFO::
testing...
2024-12-26 08:21:27,839:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:21:27,840:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:21:27,840:INFO::submit dir: submit/submit_simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:21:27,931:INFO::save_dir_name: simpleHGN_lr0.0005_wd0.0001
2024-12-26 08:21:27,945:INFO::{'micro-f1': 0.6971122797306024, 'macro-f1': 0.6524829122067338}
2024-12-26 08:21:27,989:INFO::############### Experiments Stage Ends! ###############
2024-12-26 08:21:27,989:INFO::=============== one experiment stage finish, use 401.2978858947754 time.
/home/yyj/MDNN-AC/AutoAC-main/data/IMDB
/home/yyj/MDNN-AC/AutoAC-main/data/IMDB
(3202,) (3202, 5)
/home/yyj/MDNN-AC/AutoAC-main/submit/submit_simpleHGN_lr0.0005_wd0.0001/IMDB_1.txt
multi
[93mWarning: If you want to obtain test score, please submit online on biendata.[0m
(3202,) (3202, 5)
/home/yyj/MDNN-AC/AutoAC-main/submit/submit_simpleHGN_lr0.0005_wd0.0001/IMDB_1.txt
multi
[93mWarning: If you want to obtain test score, please submit online on biendata.[0m
(3202,) (3202, 5)
/home/yyj/MDNN-AC/AutoAC-main/submit/submit_simpleHGN_lr0.0005_wd0.0001/IMDB_1.txt
multi
[93mWarning: If you want to obtain test score, please submit online on biendata.[0m
(3202,) (3202, 5)
/home/yyj/MDNN-AC/AutoAC-main/submit/submit_simpleHGN_lr0.0005_wd0.0001/IMDB_1.txt
multi
[93mWarning: If you want to obtain test score, please submit online on biendata.[0m
(3202,) (3202, 5)
/home/yyj/MDNN-AC/AutoAC-main/submit/submit_simpleHGN_lr0.0005_wd0.0001/IMDB_1.txt
multi
[93mWarning: If you want to obtain test score, please submit online on biendata.[0m
