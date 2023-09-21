***!!This is my project made as part of the training in Yandex.Practicum!!***

# gold_recovery_from_gold-bearing_ore

Prepare a prototype machine learning model for Digital. The company develops solutions for the efficient operation of industrial plants.
The model must predict the recovery rate of gold from gold-bearing ore. We have data with mining and refining parameters at our disposal.
The model will help to optimize production so that you don't run a plant with unprofitable characteristics.

Let us describe each stage:<br>
**1. Flotation . **<br>
A mixture of gold-bearing ore is fed into the flotation unit. After enrichment we get rough concentrate and "tailings", i.e. product residues with low concentration of valuable metals.
The stability of this process is affected by the unstable and non-optimal physical and chemical state of the flotation pulp (mixture of solid particles and liquid).<br>
**2. purification**<br>
The crude concentrate undergoes two purifications. The output is the final concentrate and new tailings.<br>


```
Data description:

Rougher feed - feedstock
Rougher additions (or reagent additions) - flotation reagents: Xanthate, Sulphate, Depressant
Xanthate ** - xanthogenate (promoter or activator of flotation);
Sulphate - sulphate (sodium sulphide in this production);
Depressant - depressant (sodium silicate).
Rougher process - flotation
Rougher tails
Float banks - flotation plant
Cleaner process
Rougher Au - is a rough diamond concentrate
Final Au - final gold concentrate
Parameters of the stages
air amount - air volume
Fluid levels
Feed size - size of pellets
feed rate - feed rate
```
