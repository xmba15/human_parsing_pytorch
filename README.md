# Pytorch Implementation of Human Parsing with [PSPNet](https://arxiv.org/pdf/1612.01105.pdf) on [LIP-Look Into Person-dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Gong_Look_Into_Person_CVPR_2017_paper.pdf) #
***

## Dataset ##
***

### Color Label Chart ###
***

![Color Label Chart](./docs/images/lip_color_chart.png)

## Test Result ##
<table>
    <tr>
        <td>Steven Spielberg</td>
        <td>Parsing Result</td>
    </tr>
    <tr>
        <td valign="top"><img src="docs/images/steven_spielberg.jpg"></td>
        <td valign="top"><img src="docs/images/result.jpg"></td>
    </tr>
</table>

The approach in this github repository only utilizes pspnet for training a human parsing model, without the *Self-supervised Structure-sensitive Learning* skeme in [1]. Also, the training has been done with only a single gpu (with gradient accumulation). Therefore, the parsing result is not so good.

# References
***

[1]. [Look into Person: Self-supervised Structure-sensitive Learning and A New Benchmark for Human Parsing", CVPR 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Gong_Look_Into_Person_CVPR_2017_paper.pdf)

[2]. [Pyramid Scene Parsing Network](https://arxiv.org/pdf/1612.01105.pdf)
