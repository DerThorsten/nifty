import nifty

margin = [10,10,10]
blocking = nifty.tools.blocking(roiBegin=(0,0,0), roiEnd=(100,100,100), blockShape=(10,10,10))



print(blocking.roiBegin)
print(blocking.roiEnd)
print(blocking.blockShape)
print(blocking.blockShift)

blockWithHalo = blocking.getBlockWithHalo(0, margin)
block = blocking.getBlock(0)

outerBlock = blockWithHalo.outerBlock
innerBlock = blockWithHalo.innerBlock
innerBlockLocal = blockWithHalo.innerBlockLocal




print("B ",block.begin, block.end)
print("O ",outerBlock.begin, outerBlock.end)
print("I ",innerBlock.begin, innerBlock.end)
print("IL",innerBlockLocal.begin, innerBlockLocal.end)