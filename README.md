# Reconstructing memories

Code for the paper "Bayesian reconstruction of memories stored in neural
  networks from their connectivity" ([arXiv:2105.07416](https://arxiv.org/abs/2105.07416))

## Requirements

- numpy (>= 1.8)
- scipy (tested with version 1.0.0)
- cython (tested with version 0.28.2) 

## Install

To get started, running AMP on the Hopfield model with standard binary prior and
three patterns looks like this:
```
python amp_hf.py --prior 1 -P 3
```
Now try moving this up to 16 patterns... it's pretty slow!

This is where the meanfield prior comes in. It is implemented in Cython for optimal
speed reasons. To compile, simply type
```
python setup.py build_ext --inplace
```
and run amp_hf.py with the meanfield flag:
```
python amp_hf.py --prior 1 -P 16 --mf
```
Enjoy!

## Testing

To run all tests, go to the root directory (the one that contains setup.py) and
simply type

``` 
nose2
```

## Detailed guide to the files

| File                          | Description                                                                                                                                                    |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```amp.py``` | Generic implementation of approximate message passing (AMP) <br/> for low-rank matrix factorisation                               |
| ```amp_hf*.py```                | Launches AMP for various hopfield models |
| ```cpriors.pyx```         | Cython implementation of various computationally intensive prior functions             |
| ```hopfield.py```       | Helper functions to create and analyse Hopfield networks |
| ```priors.py```           | Implementation of the various papers analysed in the paper |
| ```se_*.py```              | State evolution for the AMP algorithms |
| ```tests```             | Contains unit tests for the various modules |

# References and acknowledgements

## Approximate Message passing

The Approximate Message Passing algorithm equations are based on work by
[Thibault Lesieur] (TL), [Florent Krzakala] (FK) and [Lenka Zdeoborova] (LK),
see:

- TL, FK, and LZ, J. Stat. Mech. Theory Exp. 2017, 73403
  (2017). [arXiv:1701.00858](https://arxiv.org/abs/1701.00858)
- TL, FK and LZ, in IEEE Int. Symp. Inf. Theory - Proc. (2015),
  pp. 1635–1639.[arXiv:1503.00338](http://arxiv.org/abs/1503.00338)
- TL, FK, and LZ, in 2015 53rd
  Annu. Allert. Conf. Commun. Control. Comput. (IEEE, 2015), pp. 680–687.
  [arXiv:1507.03857](http://arxiv.org/abs/1507.03857)

They follow from earlier works by

- Yash Deshpande and Andrea Montanari ([arXiv:1402.2238](http://arxiv.org/abs/1402.2238)),
- Alyson K. Fletcher and Sundeep Rangan ([arXiv:1202.2759](http://arxiv.org/abs/1202.2759)),
- as well as [Ryosuke Matsushita and Toshiyuki
  Tanaka](http://papers.nips.cc/paper/5074-low-rank-matrix-reconstruction-and-clustering-via-approximate-message-passing)
  
### Alternative implementations of AMP algorithms

- TL, FK and LZ provided implementations for
  [Julia](https://github.com/krzakala/LowRAMP_julia) and
  [Matlab](https://github.com/krzakala/LowRAMP).
- See also the numerous implementations of AMP and related algorithms by [Phil
  Schniter](http://www2.ece.ohio-state.edu/~schniter/research.html).


# Acknowledgements

We would like to thank the Department of Mathematics at Duke University for
their hospitality during an extended visit, during which this work was carried
out. It is a pleasure to thank [Andre Manoel] for valuable discussions.

[Thibault Lesieur]: https://lesieurthibault.wordpress.com/
[Florent Krzakala]: http://www.krzakala.org/
[Lenka Zdeoborova]: http://artax.karlin.mff.cuni.cz/~zdebl9am/
[Romain Couillet]: http://romaincouillet.hebfree.org/
[Hafiz Tiomoko Ali]: http://www.laneas.com/hafiz-tiomoko-ali
[Andre Manoel]: http://ndrm.nl/


