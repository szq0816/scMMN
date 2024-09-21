import matplotlib.pyplot
import xlrd
import matplotlib.pyplot as plt

# 调节字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

dset1 = 'FLICKR'
dset2 = 'NUSWIDE'
dset3 = 'IAPR'
code_len = '128'

fig = plt.figure(figsize=(14, 5), dpi=300)


for idx, dset in enumerate(['FLICKR', 'NUSWIDE', 'IAPR']):
    plt.subplot(1, 3, (idx + 1))
    for j, task in enumerate(['I2T', 'T2I']):
        x = []
        y = []
        dataset = f'{dset}-{code_len}-Convergence_Analysis.xls'

        ## FLICKR
        data1 = xlrd.open_workbook(dataset)
        tableI2T_FLICKR = data1.sheets()[j]

        ## FLICKR
        x_data_FLICKR = tableI2T_FLICKR.row_values(0)
        y_data_FLICKR = tableI2T_FLICKR.row_values(1)
        for i in range(len(x_data_FLICKR)):
            if (i + 1) % 4 == 1:
                x.append(x_data_FLICKR[i])
                y.append(y_data_FLICKR[i])
        plt.plot(x, y, marker='o', linestyle='dashdot', linewidth=2)
        # plt.plot(x_data_FLICKR, y_data_FLICKR, marker='o', linestyle='dashdot', linewidth=2)

    if dset == 'FLICKR':
        dataset_name = 'MIR Flickr'
        plt.ylim(0.8, 0.904)  # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    elif dset == 'NUSWIDE':
        dataset_name = 'NUS-WIDE'
        plt.ylim(0.735, 0.87)  # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    else:
        dataset_name = 'IAPR TC-12'
        # plt.ylim(0.66, 0.72)  # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

    plt.title(f'{dataset_name} @{code_len} bits', verticalalignment='center',
              font={"family": "Times New Roman", "size": 16})
    plt.xlabel(f'epochs', font={"family": "Times New Roman", "size": 16})
    plt.ylabel('mAP', font={"family": "Times New Roman", "size": 16})
    plt.legend(['I2T', 'T2I'], loc='lower right', prop={"family": "Times New Roman", "size": 16})
plt.tight_layout()
plt.savefig(f'convergence.pdf')
plt.show()
plt.clf()

