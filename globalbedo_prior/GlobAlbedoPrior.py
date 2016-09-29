#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Globalbedo prior generation
============================

This script produces the GlobAlbedo prior from MODIS MCD43 data.
The prior is calculated in two stages: a first, single date
stage, which provides a weighted average of the kernel weights
for a particular day and a second stage that performs some temporal
smoothing and interpolates the result to daily. Two versions of the
prior are produced: a snow and a snow-free version.

"""
import argparse
import sys
import os
import calendar
import logging
import time

try:
    import numpy as np
except ImportError:
    print "You need to have numpy installed!"

try:
    from osgeo import gdal
except ImportError:
    print "You need to have the GDAL Python bindings installed!"
import matplotlib.pyplot as plt
from ga_utils import *


gdal.SetConfigOption("GDAL_CACHEMAX", "6400")


# Authors etc
__author__ = "P Lewis & J Gomez-Dans (NCEO&UCL)"
__copyright__ = "(c) 2014"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "J Gomez-Dans"
__email__ = "j.gomez-dans@ucl.ac.uk"
__status__ = "Development"


# MAGIC number
MAGIC = 0.61803398875
gdal_opts = [ "COMPRESS=LZW", "INTERLEAVE=BAND", "TILED=YES", "BIGTIFF=YES" ]

# Set up logging
LOG = logging.getLogger( __name__ )
OUT_HDLR = logging.StreamHandler( sys.stdout )
OUT_HDLR.setFormatter( logging.Formatter( '%(asctime)s %(message)s') )
OUT_HDLR.setLevel( logging.INFO )
LOG.addHandler( OUT_HDLR )
LOG.setLevel( logging.INFO )

class GlobAlbedoPrior ( object ):
    """
    """
    def __init__ ( self, tile, data_dir, output_dir, bands=[1,2,3,4,5,6,7], no_snow=True ):
        """This configures where the data are stored, the tile and the output
        directory. It also allows the choice of bands to process, which by
        default is the 7 MODIS bands.

        Parameters
        ----------
        tile: str
            The tile in MODIS-speak, e.g. "h17v04"
        data_dir: str
            The top directory where input data are stored
        output_dir: str
            The directory where the output will be saved to
        bands: array-like
            The set of bands to be processed. By default, all 7 bands.
        no_snow: bool
            A flag indicating whether to choose the snow free or the snow albedo results
        """
        self.tile = tile
        LOG.info ("Doing tile %s" % self.tile )
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.bands = bands
        LOG.info ("Doing bands %s" % str ( self.bands ))

        # Now, get the list of filenames that we are going to use...
        a1, a2 = self.get_modis_fnames ()

        LOG.info ("Found %d(MCD43A1)/%d(MCD43A2) HDF files in %s" % \
            ( a1, a2, self.data_dir ) )
        LOG.info ("Will write output in %s" % self.output_dir )
        self.no_snow = no_snow

    def get_modis_fnames ( self ):
        """Grab MODIS filenames
        This method gets the MODIS filenames and stores them in dictionaries
        that can be accessed by DoY. This is quite useful for Stage 1 prior
        creation. This method scans `self.data_dir` for MCD43A1/MCD43A2 files
        using a filename mask. It only does Collection 5.
        """
        self.fnames_mcd43a1 = {}
        self.fnames_mcd43a2 = {}
        a1_files = 0
        a2_files = 0
        for doy in xrange ( 1, 367, 8 ):
            pattern = "MCD43A1.A????%03d.%s.005.*.hdf" % ( doy, self.tile )
            self.fnames_mcd43a1[doy] = [ f for f in locate( pattern, \
                root=self.data_dir ) ]
            pattern = "MCD43A2.A????%03d.%s.005.*.hdf" % ( doy, self.tile )
            self.fnames_mcd43a2[doy] = [ f for f in locate( pattern, \
                root=self.data_dir ) ]
            a1_files += len ( self.fnames_mcd43a1[doy] )
            a2_files += len ( self.fnames_mcd43a2[doy] )
        if a1_files != a2_files:
            raise IOError("I didn't find the same number of MCD43A1 and MCD43A2 files!")
        if a1_files == 0:
            raise IOError("No files found!!! You sure about the tile and directory?")
        return a1_files, a2_files

    def _interpret_qa ( self, qa_data ):
        """Interpret QA
        A method to interpret the QA according to the GlobAlbedo specs. The
        QA flags are translated into a numerical value using `MAGIC`, which
        is by default defined to be the golden proportion.

        Parameters
        ----------
        qa_data: array
            An n_years*nx*ny array of QA data, from the MODIS product

        Returns
        --------
        An array with the relevant calculated weights
        """
        # First stage: consider the QA flag, first 3 bits
        qa = np.bitwise_and ( qa_data, 7L ) # 7 is b000111, ie get the last 3 bits
        weight = np.where ( qa < 4, MAGIC**qa, 0. )
        return weight

    def _interpret_snowmask ( self, snow_data ):
        """Interpret the snow mask
        Interpret the snow mask as a boolean array, depending on whether we are
        doing the snow or the snowfree prior.

        Returns
        --------
        A snow mask (boolean)
        """

        if self.no_snow:
            snow = np.where ( snow_data == 0, True, False ) # Snow free
        else:
            snow = np.where ( snow_data == 1, True, False ) # Snow
        return snow

    def _interpret_landcover ( self, landcover ):
        """Interpret the landcover information
        The landcover flags are filtered. In principle, we just choose
        everything under 7, but this can obviously be refined.

        Returns
        A boolean array of suitable/unsuitable landcover type
        """
        # Landcover uses bits 04-07, so 8 + 16 + 32 + 64 = 120L
        # then shift back 3 positions
        mask = np.right_shift(np.bitwise_and ( landcover, 120L), 3)
        mask = np.where ( mask < 7, True, False ) # ignore deep ocean
        return mask

    def create_output ( self ):
        self.output_ptrs = {}

        for band in self.bands:
            for doy in self.fnames_mcd43a1.iterkeys():
                for kernel in [ "iso", "volu", "geo"]:
                    output_fname = os.path.join ( self.output_dir, \
                        "MCD43P.%03d.%s.%s.b%d.tif" % \
                        ( doy, kernel, self.tile, band ) )
                    LOG.debug ( "Creating %s" % output_fname )


                    if os.path.exists ( output_fname ):
                        LOG.debug( "Removing %s... " % output_fname )
                        os.remove ( output_fname )
                    drv = gdal.GetDriverByName ( "GTiff" )
                    drv.Register()
                    output_prod = "%03d_%s_b%d" % ( doy, kernel, band )
                    self.output_ptrs[output_prod] = drv.Create( output_fname, \
                        2400, 2400, 2, gdal.GDT_Float32, options=gdal_opts )
                    LOG.debug("\t... Created!")

    def do_qa ( self, data_in, n_years ):
        """A simple method to do the QA from the read data. The reason for this is
        to get Python's garbage collector to deallocate the memory of all these
        temporary arrays after we have calculated the mask. We assume that
        `data_in` stores `n_years` of BRDF data, QA data, snow data and
        ancillary data. The order is important, as we use the positions in
        `data_in` to figure out what each array is.

        Parameters
        -----------
        data_in: list
            A list with the different data: kernels, QA, snow,
            ancillary information...
        n_years: int
            The number of years.

        Returns
        -------
        A mask of interpreted QA, with 0 where data are missing.
        """

        qa_data = np.array ( data_in [ (n_years):(2*n_years)] )
        snow_data = np.array ( data_in [ (2*n_years):(3*n_years)] )
        land_data = np.array ( data_in[ (3*n_years):] )

        qa = self._interpret_qa ( qa_data )
        snow = self._interpret_snowmask ( snow_data )
        land = self._interpret_landcover ( land_data )
        mask = qa*snow*land
        return mask


    def stage1_prior ( self ):
        """Produce the stage 1 prior, which is simply a weighted average of the
        kernel weights. This"""

        self.create_output ()
        for band in self.bands:
            for doy in xrange ( 1, 367, 8 ):
                LOG.info ("Doing band %d, DoY %d..." % ( band, doy ) )
                obs_fnames = [ 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band%d' % ( f, band ) \
                        for f in self.fnames_mcd43a1[doy] ]
                qa_fnames = [ 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Band_Quality' % ( f ) \
                        for f in self.fnames_mcd43a2[doy] ]
                snow_fnames = [ 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:Snow_BRDF_Albedo' % ( f ) \
                        for f in self.fnames_mcd43a2[doy] ]
                land_fnames = [ 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Ancillary' % ( f ) \
                        for f in self.fnames_mcd43a2[doy] ]


                all_files = obs_fnames + qa_fnames + snow_fnames + land_fnames
                for (ds_config, this_X, this_Y, nx_valid, ny_valid, data_in )  \
                        in extract_chunks ( all_files ):
                    LOG.info ("\t\tRead a %d x %d pixel chunk..." % ( ny_valid, nx_valid ) )
                    n_years = len ( data_in )/4 # First lot will be BRDF parameters, 2nd will be QA,
                                                # 3 snow mask and fourth land mask

                    mask = self.do_qa ( data_in, n_years )
                    brdf_data = np.array ( data_in [ :n_years] )
                    data_in = None # Clear memory a bit
                    # Process per kernel weight
                    prior_mean, prior_var = calculate_prior ( brdf_data, mask )
                    # data_in contains all the data. Half the samples bands are
                    # BRDF parameters
                    for i, kernel in enumerate ( [ "iso", "volu", "geo" ] ):
                        output_prod = "%03d_%s_b%d" % ( doy, kernel, band )
                        bout = self.output_ptrs[output_prod].GetRasterBand ( 1 )
                        bout.SetNoDataValue(-999.0)
                        bout.WriteArray ( prior_mean[i, :, :].astype(np.float32), \
                            xoff=this_X, yoff=this_Y )
                        bout.FlushCache()
                        del bout
                        bout = self.output_ptrs[output_prod].GetRasterBand ( 2 )
                        bout.SetNoDataValue(-999.0)
                        bout.WriteArray ( prior_var[i, :, :].astype(np.float32), \
                            xoff=this_X, yoff=this_Y )
                        bout.FlushCache()
                        del bout
                # close files
                for kernel in [ "iso", "volu", "geo" ]:
                    output_prod = "%03d_%s_b%d" % (doy, kernel, band)
                    self.output_ptrs[output_prod].SetGeoTransform( ds_config['geoT'] )
                    self.output_ptrs[output_prod].SetProjection( ds_config['proj'] )
                    # close the file
                    self.output_ptrs[output_prod] = None
                LOG.info( "\t\t Burrpp!!" )

    def stage2_prior( self ):
        t = np.arange ( 1, 367 )
        ts = np.arange ( 1, 367, 8 )
        w = np.exp ( -0.0866*np.abs ( t - 183) )
        w = np.roll ( w, -183 ) # Shift to 31 Dec
        for band in self.bands:
            do the weighted means
            prod = 'mean'
            for kernel in [ "iso", "volu", "geo"]:
                drv = gdal.GetDriverByName ( "GTiff" )
                output_fname = os.path.join ( self.output_dir, \
                   "MD43P.%s.%s.%s.b%d.tif" % ( prod, kernel, self.tile, band ) )
                dst_ds = drv.Create ( output_fname,  2400, 2400, len(t), \
                   gdal.GDT_Float32, options=gdal_opts )

                mean_vals = np.empty ((len(ts), 2400, 2400))
                i = 0
                for d in ts:
                    g = gdal.Open ( os.path.join ( self.output_dir, \
                        "MCD43P.%03d.%s.%s.b%d.tif" % \
                        ( d, kernel, self.tile, band ) ))
                    mean_vals [ i, :, : ]= g.GetRasterBand(1).ReadAsArray()
                    i += 1
                # make into masked array
                mean_vals = np.ma.array ( mean_vals, mask=mean_vals == 0.)
                # iterate over days
                for this_doy in t:
                    LOG.info( "Computing %s for doy %i on band %i, %s kernel" % (prod, this_doy, band, kernel) )
                    w_8day = np.roll ( w, this_doy )[ts]
                    mean_interp = np.ma.average ( mean_vals, \
                        axis=0, weights=w_8day )
                    dst_ds.GetRasterBand ( this_doy ).WriteArray ( mean_interp )
                dst_ds.SetGeoTransform ( g.GetGeoTransform() )
                dst_ds.SetProjection ( g.GetProjection() )
                dst_ds = None
            """
            Do the weighted uncertainties
            ------------------------------

            This is unfortunately more involved.

            We assume that each set of kernel weights at 8 days are Gaussian with

            f_i = G(mean, sigma)

            Therefore to make a daily weighted Gaussian we need to weight
            the means which is easy but also the variances which being the second
            moment is dependent on the mean.


            if we define f(x) as the weighted product according to weights w:

            the mean is

            E[X] = f(x) = \sum_{i}^{N} = w_i mu_i

            considering the variance is:

            sigma_{i}^{2} =  E[x^2] - (E[X])^2

            the weighted variance is


            var(f) = \sum_{i} w_i * \sigma_{i}^{2} +  \sum_{i} w_i * (\mu_i)^{2}
                    - \left( \sum_{i} w_i \mu_i   \right)^{2s}


            the subtraction is a correction term for the different means

            (see http://stats.stackexchange.com/questions/
                    16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians)

            """
            # do the weighted means
            prod = 'var'
            for kernel in [ "iso", "volu", "geo"]:
                drv = gdal.GetDriverByName ( "GTiff" )
                output_fname = os.path.join ( self.output_dir, \
                   "MD43P.%s.%s.%s.b%d.tif" % ( prod, kernel, self.tile, band ) )
                dst_ds = drv.Create ( output_fname,  2400, 2400, len(t), \
                   gdal.GDT_Float32, options=gdal_opts )

                mean_vals = np.empty ((len(ts), 2400, 2400))
                var_vals = np.empty ((len(ts), 2400, 2400))
                i = 0
                for d in ts:
                    g = gdal.Open ( os.path.join ( self.output_dir, \
                        "MCD43P.%03d.%s.%s.b%d.tif" % \
                        ( d, kernel, self.tile, band ) ))
                    mean_vals [ i, :, : ]= g.GetRasterBand(1).ReadAsArray()
                    var_vals [ i, :, : ]= g.GetRasterBand(2).ReadAsArray()
                    i += 1
                # make into masked arrays
                mean_vals = np.ma.array ( mean_vals, mask=mean_vals == 0.)
                var_vals = np.ma.array ( var_vals, mask=mean_vals == 0.)
                # loop over days t
                for this_doy in t:
                    LOG.info( "Computing %s for doy %i on band %i, %s kernel" % (prod, this_doy, band, kernel) )
                    w_8day = np.roll ( w, this_doy )[ts]
                    """
                    Calculate weighted variances
                    \sum_{i} w_{i} * \sigma_{i}^{2}
                    """
                    weighted_variances = np.ma.average ( var_vals, \
                        axis=0, weights=w_8day )
                    """
                    Calculate weighted square means
                    \sum_{i} w_{i} * (\mu_{i})^{2}
                    """
                    weighed_squared_means = np.ma.average ( mean_vals**2, \
                        axis=0, weights=w_8day )
                    """
                    calculate square weigthed means
                    ( \sum_{i} w_{i} \mu_{i}  )^{2}
                    """
                    squared_weighted_means = np.ma.average ( mean_vals, \
                        axis=0, weights=w_8day )**2
                    """
                    total weighted variance:
                    ------------------------
                    var(f) = \sum_{i} w_{i} * \sigma_{i}^{2} + \sum_{i} w_{i} * (\mu_{i})^{2}
                             -  ( \sum_{i} w_{i} \mu_{i}  )^{2}
                    """
                    total_weighted_variance = ( weighted_variances +
                                                weighed_squared_means
                                                - squared_weighted_means )
                    # save
                    dst_ds.GetRasterBand ( this_doy ).WriteArray ( total_weighted_variance )
                dst_ds.SetGeoTransform ( g.GetGeoTransform() )
                dst_ds.SetProjection ( g.GetProjection() )
                dst_ds = None

