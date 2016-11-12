# Data Flow Across the Modules

These pseudo data flow diagrams show how the data flow from input through the three modules of the implementation. These are high level overview of the internal operation of the modules that should ease the process of reading into the source code.

**Notation**
* **B**: the batch size.
* **T**: the sequence length.
* **the rest** follows the paper notation.

*No data flow diagram is shown for the memory module as it would be unnecessarily complicated; given that , unlike the other two modules, the memory model is not doing anything more than what is described in the paper*.

## DNC Module

![DNC-pDFD](/assets/DNC-DFD.png)

## Controller Module

![Controller-pDFD](/assets/Controller-DFD.png)
