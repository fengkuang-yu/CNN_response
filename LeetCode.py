predict_loop = [96]  # 选取不同的检测线圈进行预测
pred_intervals = [0]  # 预测的时间长度

time_lag_initial = 4  # 预测时间变化
space_initial = 4  # 预测使用的检测线圈数目

params = Parameters()

def cnn_train(data):
    pass
def train_test_data(s, t):
    pass

def STFSA(param, step=4, cur_space=4, cur_time=4, eps=4):
    opt_time = cur_time
    opt_space = cur_space
    while cur_space - opt_space < eps:
        data = train_test_data(cur_space, cur_space)
        if cnn_train(data) < cnn_train(data):
            opt_space = cur_space
        else:
            cur_space += step
    while cur_time - opt_time < eps:
        data = train_test_data(cur_time, cur_space)
        if cnn_train(data) < cnn_train(data):
            opt_time = cur_time
        else:
            cur_time += step
    return [opt_space, opt_time], cnn_train(train_test_data())


for cur_loop in predict_loop:
    params.predict_loop = cur_loop
    for cur_intervals in pred_intervals:
        params.predict_intervals = cur_intervals
        with open(os.path.join(params.file_path,
                               'model\\loop{}_res_error{}.csv'.format(params.predict_loop,
                                                                      (1 + params.predict_intervals) * 5)),
                  'w', newline='') as csvfile:
            fieldnames = ['pred_interval', 'space_time', 'MAPE', 'MAE']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        for loop_num in space:
            for time_intervals in time_lag:
                params.loop_num = loop_num
                params.time_intervals = time_intervals
                Data = train_test_data(params)
                mape, mae = train(Data, params)
                with open(os.path.join(params.file_path,
                                       'model\\loop{}_res_error{}.csv'.format(params.predict_loop,
                                                                              (1 + params.predict_intervals) * 5)),
                          'a+', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    time_space = 'space{}*time{}'.format(params.loop_num, params.time_intervals)
                    writer.writerow({'pred_interval': (1 + params.predict_intervals) * 5,
                                     'space_time': time_space, 'MAPE': mape, 'MAE': mae})
                counter += 1
                print('COMPLETED {val:.2%}'.format(val=(counter / total)))
