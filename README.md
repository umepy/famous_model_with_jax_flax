# beggining Computer Vision with Flax

CVFlax is a material to study Computer Vision models using Flax


## Models

1. **AlexNet**
2. **VGG**
3. **ResNet**

## Dataset infomation

We uses `Food-101` dataset.

### Normalization parameter

- mean: `[0.54498774 0.4434933  0.34360075]`
- std: `[0.23354167 0.24430245 0.24236338]`

## Implementation memo

- Convのstrideは`int`ではなく、縦方向と横方向分で`Tuple(int,int)`で渡す
- Convのpaddingは`int`ではなく、上下左右の幅として`Tuple(Tuple(int,int),Tuple(int,int))`で渡す
- Convのpaddingはデフォルトで0じゃないのでしっかり引数を渡す
