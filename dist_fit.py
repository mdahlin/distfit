import pandas as pd
import numpy as np
import scipy.stats as st 
import matplotlib.pyplot as plt

#TODO: saving functionality (plots, fit dists, tables, etc.)
#TODO: pep8 formatting
#TODO: error and warning handeling
#TODO: discrete distribution support
#errm_standard
class UnivariateDistFit:
    """
    a class to perform a standard univariate distirbution fitting process on provided data.

    ...
    
    Attributes
    ----------
    DISTRIBUTIONS : dict
        dictionary of all possible distributions to fit the data to
    filteredDistributions : list
        list of distribtions after any filtering applied when initializing the class
    data : array_like
        a 1 x N dimensional array or list containing the data to be fit
    fitDF : DataFrame
        a dataframe containing statistics (columns) for each of the distirbutions fit 
    
    Methods
    -------
    printEmpSummary():
        prints summary statistics related to the data supplied
    printFit(pretty_print=False, col_fmt = '{:.3}'):
        prints the fitDF attribute
    plotFit(plotly=False):
        plots empirical data and pdfs of fitted distributions
    plotCompare(metric='BIC', top_n=10, gof_filter=None, gof_alpha=0.05):
        plots a comparison between the different distirbutions
    runAll(pretty_print=False, metric='BIC'):
        runs all pieces of the class to allow for a quick one-line for analysis
    """

    DISTRIBUTIONS = {
        #TODO: add more distributions/characteristics
        #currently commented out the dists resulting in warnings
        st.argus: ['argus', 'continuous', '01_ni support', '1param'],
        st.beta: ['beta', 'continuous', '01_i support', '2param'],
        st.bradford: ['bradford', 'continuous', '01_i support', '1param'],
        #st.burr: ['burr', 'burr3', 'continuous', '0inf_i support'],
        #st.burr12: ['burr12', 'continuous', '0inf_i support', '2param],
        st.cauchy: ['cauchy', 'continuous', 'real support', '2param'],
        #st.chi2: ['chi2', 'continuous', '0inf_ni support', '1param'],
        st.dgamma: ['dgamma', 'continuous', 'real support'],
        st.dweibull: ['dweibull', 'continuous', 'real support'],
        st.expon: ['expon', 'continuous', '0inf_i support', '1param'],
        st.exponnorm: ['exponnorm', 'continuous', 'real support', '3param'],
        st.exponweib: ['exponweib', 'continuous', '0inf_ni support'], 
        #st.f: ['f', 'continuous', '0inf_ni support'],
        #st.fatiguelife: ['fatiguelife', 'continuous', '0inf_i support'], probably won't even include
        st.fisk: ['fisk', 'continuous', '0inf_i support'],
        st.genextreme: ['genextreme', 'continuous', 'real support'],
        st.gamma: ['gamma', 'continuous', '0inf_i support', '2param'],
        st.gompertz: ['gompertz', 'continuous', '0inf_i support'],
        st.gumbel_l: ['gumbel_l', 'continuous', 'real support'],
        st.gumbel_r: ['gumbel_r', 'continuous', 'real support'],
        #st.halfcauchy: ['halfcauchy', 'continuous', '0inf_i support'],
        #st.halflogistic: ['halflogistic', 'continuous', '0inf_i support'],
        #st.halfnorm: ['halfnorm', 'continuous', '0inf_i support'],
        #st.invgamma: ['invgamma', 'continuous', '0inf_i support']
        st.invgauss: ['invgauss', 'continuous', '0inf_i support'],
        #st.invweibull: ['invweibull', 'continuous', '0inf_ni support']
        st.laplace: ['laplace', 'continuous', 'real support', '2param'],
        st.logistic: ['logistic', 'continuous', 'real support', '2param'],
        st.loggamma: ['loggamma', 'continuous', 'real support'],
        #st.loglaplace: ['loglaplace', 'continuous', 'real support'],
        st.lognorm: ['lognorm', 'continuous', 'real support', '2param'],
        st.loguniform: ['loguniform', 'continuous', 'real support'],
        #st.lomax: ['lomax', 'continuous', '0inf_i support'].
        st.maxwell: ['maxwell', 'continuous', '0inf_i support'],
        st.moyal: ['moyal', 'continuous', 'real support', '2param'],
        st.norm: ['norm', 'continuous', 'real support', '2param'],
        #st.norminvgauss: ['norminvgauss', 'continuous', 'real support']
        #st.pareto: ['pareto', 'continuous', '1inf_i support']
        #st.powerlaw: ['powerlaw', 'continuous', '01_i support'],
        #st.powerlognorm: ['powerlognorm', 'continuous', 'real support'],
        #st.powernorm: ['powernorm', 'continuous'. 'real support'],
        st.rayleigh: ['rayleigh', 'continuous', '0inf_i support'],
        #st.recipinvgauss: ['recipinvgauss', 'continuous', '0inf_i support', '2param'],
        st.t: ['t', 'continuous', 'real support', '3param'],
        st.uniform: ['uniform', 'continuous', 'real support'],
        st.weibull_min: ['weibull_min', 'continuous', '0inf_ni support'],
        st.weibull_max: ['weibull_max', 'continuous', '0inf_ni support'],
        st.wald: ['wald', 'continuous', '0inf_i support']
    }
    
    characteristics = list({ele for val in DISTRIBUTIONS.values() for ele in val[1:]})
    #TODO: add additional helpful stats
    def _calcFit(self, dist, data):
        """fits the distirbution to the data and computes other stats"""
        distName = dist.__class__.__name__.split('_gen')[0]
        params = dist.fit(data)
        paramsFull = (list(params) + [np.nan, np.nan, np.nan])[0:4]
        logLik = dist.logpdf(data, *params).sum()
        ksPvalue = st.kstest(data, distName, params)[1]
        bic = len(params) * np.log(len(data)) - 2 * logLik
        p99 = dist.ppf(.99, *params)
        p01 = dist.ppf(.01, *params)

        outDF = pd.DataFrame([[distName] + paramsFull 
                            + [logLik] + [ksPvalue] + [bic] 
                            + [p01] + [p99]],
                columns=['dist', 'Param1', 'Param2', 'Param3', 'Param4',
                        'logLik', 'ksPvalue', 'BIC', 'P01', 'P99'])
        return outDF

    
    def __init__(self, data, filters='all', filter_type='any'):
        """
        initilizes the class, filters down the pool of distirbutions to use,
        and fits distributions/calcs stats

        Parameters
        ----------
        data : array_like
            a 1 x N dimensional array or list containing the data to be fit
        filters : 'all' or list, optional
            if 'all' (default), then all distributions defined in DISTRIBUTIONS will be used
            if a list, then list should contain characteristics to filter on
            will raise an error which will include possible characteristics if filtered incorrectly
        filter_type : {'any', 'all'}, optional
            describes the filtering method to select distributions based on the filter argument
            'any' (default) will select any distribution where at least 1 characteristic is met
            'all' will select only distributions where all characterisitcs are met
        """
        self.data = data
        
        #### FILTER DOWN THE DISTRIBUTIONS ####
        self.filteredDistributions = []
        if filters == 'all':
            self.filteredDistributions = [val for val in self.DISTRIBUTIONS.keys()]
        else:
            for dist, characterisics in self.DISTRIBUTIONS.items():
                if filter_type == 'any':
                    if any((f in characterisics) is True for f in filters):
                        self.filteredDistributions.append(dist)
                elif filter_type == 'all':
                    if all((f in characterisics) is True for f in filters):
                        self.filteredDistributions.append(key)

        #error incase there are no distirbutions remaining after filtering
        if len(self.filteredDistributions) == 0:
            raise ValueError('No distributions after filtering,' +
                ' try changing filter_type to "any" or double check the spelling in filters.' +
                '\n Avaialble characteristics: ' + 
                ', '.join(self.characteristics))
    
        #### FIT THE DISTRIBUTIONS AND CALC STATS ####    
        self.fitDF = pd.DataFrame(columns=['dist', 'Param1', 'Param2', 'Param3', 'Param4',
                                           'logLik', 'ksPvalue', 'BIC', 'P01', 'P99'])

        for dist in self.filteredDistributions:
            #TODO: add warning/error handeling logic here
            self.fitDF = self.fitDF.append(self._calcFit(dist, self.data))

        self.fitDF = self.fitDF.set_index('dist')
        self.fitDF = self.fitDF.sort_values('BIC')


    def printEmpSummary(self):
        """prints summary statistics of the input data used for fitting"""
        #TODO: add more relevant stats / organize
        print("\nEmpirical Summary\n", "-" * 20)
        print("# Obs --> ", len(self.data))
        print("Mean --> ", np.mean(self.data))
        print("Meadian --> ", np.median(self.data))
        print("SD --> ", np.std(self.data))
        print("Min --> ", np.min(self.data))
        print("Max --> ", np.max(self.data))
        print("1st %tile --> ", np.percentile(self.data, 1, interpolation='nearest'))
        print("99th %tile --> ", np.percentile(self.data, 99, interpolation='nearest')) 
        
    
    def printFit(self, pretty_print=False, col_fmt='{:.3f}'):
        """
        prints the results from the distibution fitting / statistics calc

        Parameters
        ----------
        pretty_print : bool, optional
            Print method. If False (default), standard DataFrame printing. If True, 
            adds cell highlighting, coloring, and formatting 
        col_fmt : string or dict, optional
            input into the format() function. By default rounds all columns to 3
            decimals. Something like `{'Param1':"{:.1}", 'Param2':"{:.1}",
            'Param3':"{:.1}", 'logLik':"{:.1}", 'ksPvalue':"{:.3}",
            'BIC':"{:.1}", 'P01':"{:.1%}", 'P99':"{:.1%}"}` could be fed in for
            showing the perentile percentages when fitting ratios

        Returns
        -------
        If pretty_print is set to `False`, returns nothing and prints the fitDF attribute
        If pretty_print is set to `True`, returns a styled DataFrame with min and max values
          highlighted in the P01 and P99 columns respectively, colors KS pvalues less than alpha red, 
          and formats columns based on col_fmt
        
        Notes
        -----
        The pretty print feature will only work in Jupyter as far as I know
        """
        if not pretty_print:
            print(self.fitDF)
        else:
            #taking from the panads docs https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
            def _highlight_max(data, color='yellow'):
                '''
                highlight the maximum in a Series or DataFrame
                '''
                attr = 'background-color: {}'.format(color)
                if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
                    is_max = data == data.max()
                    return [attr if v else '' for v in is_max]
                else:  # from .apply(axis=None)
                    is_max = data == data.max().max()
                    return pd.DataFrame(np.where(is_max, attr, ''),
                                        index=data.index, columns=data.columns)
            def _highlight_min(data, color='yellow'):
                '''
                highlight the minimum in a Series or DataFrame
                '''
                attr = 'background-color: {}'.format(color)
                if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
                    is_min = data == data.min()
                    return [attr if v else '' for v in is_min]
                else:  # from .apply(axis=None)
                    is_min = data == data.min().min()
                    return pd.DataFrame(np.where(is_min, attr, ''),
                                        index=data.index, columns=data.columns)
            def _color_alpha_red(val, alpha = .05):
                """
                Takes a scalar and returns a string with
                the css property `'color: red'` for less than alpha
                strings, black otherwise.
                """
                if val < alpha: 
                    return 'color: red'
                else:
                    pass
            
            return self.fitDF.style.\
                format(col_fmt).\
                apply(_highlight_max, subset=['P99']).\
                applymap(_color_alpha_red, subset=['ksPvalue']).\
                apply(_highlight_min, subset=['P01'])  
        return None
        
    
    def plotFit(self, bins=10, **kwargs):
        """
        plots empirical data and pdfs of fitted distributions

        Parameters
        ----------
        bins : int, optional
            specifies the bins in the histogram of empirical data
        **kwargs
            additional arguments for plt.legend 
        """
        points = np.linspace(self.data.min(), self.data.max(), 200)

        fig, ax = plt.subplots()
        # actual data 
        plt.hist(self.data, bins=bins, color='grey', density=True)
        # fit distributions
        for i in range(len(self.fitDF)):
            x = self.fitDF.iloc[i]
            dist = eval('st.' + x.name)
            params = tuple(x[0:4].dropna())
            pdf = dist.pdf(points, *params)
            label = x.name
            plt.plot(points, pdf, label=label)
        # additional cosmetics 
        #TODO: clean up output viz in the future
        plt.legend(**kwargs)

        plt.show()
        #TODO: add logic for a more interactive output
        return None
    
    
    def plotCompare(self, metric='BIC', top_n=10, gof_filter=None, gof_alpha=0.05):
        """
        plots a comparison between the different distirbutions based on metric/column from fitDF

        Parameters
        ----------
        metric : str, optional
            the metric from fitDF to plot; defaults to BIC
        top_n : int, optional
            limit to a the set number of distirbutions (default 10)
        gof_filter : {None, 'ksPvalue'}, optional
            default to no filter. If 'ksPvalue', filters out any dists
            that don't meet the alpha threshold of gof_alpha
        gof_alpha : num, optional
            an alpha threshold for if gof_filter is not None (default 0.05)
        """
        df = self.fitDF
        if gof_filter:
            df = df[df[gof_filter] > gof_alpha]
        df = df.sort_values(metric)
        df = df[0:top_n]
        plt.plot(df.index, df[metric], marker='x')
        plt.title(metric + " for each Distribution")
        plt.xticks(rotation=45)
        plt.show()
        return None
    
    
    def runAll(self, pretty_print=False, metric='BIC'):
        """
        runs all element of the class

        Parameters
        ----------
        pretty_print : bool, optional
            Print method. If False (default), standard DataFrame printing. If True, 
            adds cell highlighting, coloring, and formatting 
        plotly : bool, optional
            jkljlk
        metric : str, optional
            the metric from fitDF to plot; defaults to BIC
        """
        self.printEmpSummary()
        self.plotFit()
        print("\n")
        print("\nFit Summary\n" + "-" * 10)
        self.printFit(pretty_print=pretty_print)
        print("\n")
        self.plotCompare(metric=metric)
        return None
