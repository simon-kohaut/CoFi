"""CoFi - The Constitutional Filter for Bayesian estimation of compliant agents."""

#
# Copyright (c) Simon Kohaut, Felix Divo, and contributors
#
# This file is part of CoFi and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with CoFi.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# CoFi
from .filter import ConstitutionalFilter, FilterResult, systematic_resample

__all__ = ["ConstitutionalFilter", "FilterResult", "systematic_resample"]
__version__ = "0.0.1"
__author__ = "Simon Kohaut"
