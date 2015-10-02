### Questions:

1. Where did that product come from? What assumption does it encode?

> The product was implied by the plate (box) containing "datapoints k". The PGM illustrates the factorisation of the joint PDF for
all variables, so if the datapoints are independent, we can write
the sampling distribution for all $y_k$'s' as the simple product
of independent terms ${\rm Pr}(y_k\,|\,y^{\rm true}_k,\sigma_k)$. In general these $y_k$ will not be *identically* distributed (as the
$\sigma_k$'s might be different, for one), but by drawing a plate we are  assuming they are *independently* distributed.


2. Why are the $\{x_k\}$ and the $\{\sigma_k\}$ always on the right hand side of the bar, but $\{y_k\}$ is sometimes on the left?

> We are taking $\{x_k\}$ and $\{\sigma_k\}$ to be *constants*, *given* to us as part of the experiment design - so all our inferences
and conditional PDFs are calculated *given* these constants.

3. What is the meaning of "$H_1$"? *Hint: notice that it too is always on the right hand side of the bar.*

> TBF

4. What functional form does the conditional PDF ${\rm Pr}(y^{\rm true}_k\,|\,x_k,m,b,H_1)$ have?

> TBF

5. What functional form do you think we should assume for the sampling distribution, ${\rm Pr}(y_k\,|\,y^{\rm true}_k,x_k,\sigma_k,H_1)$? *Hint: imagine you were generating mock data.*

> TBF

6. What functional form should we assume for ${\rm Pr}(m,b\,|\,H_1)$?

> TBF
