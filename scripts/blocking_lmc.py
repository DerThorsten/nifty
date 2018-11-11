











import nifty

shape = [512, 512]

blocks = nifty.tools.blocking(
    roiBegin=(0,0),
    roiEnd=shape,
    blockShape=(64,64),
    blockShift=(0,0)
)

for block_index in range(blocks.numberOfBlocks):
    block = blocks.getBlock(block_index)

    print(block.begin, block.end)