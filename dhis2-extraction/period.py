"""Partial implementation of the DHIS2 date and period format.

See DHIS2 documentation for more info:
<https://docs.dhis2.org/en/develop/using-the-api/dhis-core-version-235/web-api.html#webapi_date_perid_format>
"""

import math
import re
import typing
from datetime import datetime
from enum import Enum

from dateutil.relativedelta import relativedelta


class Interval(Enum):
    """DHIS2 interval."""

    DAY = 1
    WEEK = 2
    MONTH = 3
    YEAR = 4
    WEEK_SUNDAY = 5
    BI_MONTH = 6
    QUARTER = 7
    SIX_MONTH = 8
    SIX_MONTH_APRIL = 9
    FINANCIAL_YEAR_APRIL = 10
    FINANCIAL_YEAR_JULY = 11
    FINANCIAL_YEAR_OCT = 12


class Period:
    def __init__(self, period: str):
        """DHIS2 period.

        See DHIS2 docs for more info:
        <https://docs.dhis2.org/en/develop/using-the-api/dhis-core-version-235/web-api.html#webapi_date_perid_format>

        Parameters
        ----------
        period : str
            DHIS2 period string.
        """
        self.period = period
        self.parse()

    def __repr__(self):
        return f"Period('{self.period}')"

    def __str__(self):
        return self.period

    def __eq__(self, other):
        if not isinstance(other, Period):
            return False
        return self.period == other.period

    def parse(self):
        """Parse DHIS2 period string."""
        if re.match(r"^\d{8}$", self.period):
            self.interval = Interval.DAY
            self.datetime = datetime.strptime(self.period, "%Y%m%d")
            self.length = relativedelta(days=1)
        # In this implementation, weekdays preceding the first monday
        # of the year are considered as Week 00.
        # TODO: Is it the same in DHIS2 ?
        elif re.match(r"^\d{4}W\d{1,2}$", self.period):
            self.interval = Interval.WEEK
            self.datetime = datetime.strptime(self.period, "%YW%W")
            self.length = relativedelta(weeks=1)
        elif re.match(r"^\d{4}SunW\d{1,2}$", self.period):
            self.interval = Interval.WEEK_SUNDAY
            self.datetime = datetime.strptime(self.period, "%YSunW%U")
            self.length = relativedelta(weeks=1)
        elif re.match(r"^\d{6}$", self.period):
            self.interval = Interval.MONTH
            self.datetime = datetime.strptime(self.period, "%Y%m")
            self.length = relativedelta(months=1)
        elif re.match(r"^\d{6}B$", self.period):
            self.interval = Interval.BI_MONTH
            self.datetime = datetime.strptime(self.period, "%Y%mB")
            self.length = relativedelta(months=2)
        elif re.match(r"^\d{4}Q\d$", self.period):
            self.interval = Interval.QUARTER
            year = int(self.period[:4])
            quarter = int(self.period[5])
            self.datetime = datetime(year, quarter * 3 - 2, 1)
            self.length = relativedelta(months=3)
        elif re.match(r"^\d{4}S\d$", self.period):
            self.interval = Interval.SIX_MONTH
            self.datetime = datetime(int(self.period[:4]), 1, 1)
            self.length = relativedelta(months=6)
        elif re.match(r"^\d{4}AprilS\d$", self.period):
            self.interval = Interval.SIX_MONTH_APRIL
            self.datetime = datetime(int(self.period[:4]), 4, 1)
            self.length = relativedelta(months=6)
        elif re.match(r"^\d{4}$", self.period):
            self.interval = Interval.YEAR
            self.datetime = datetime(int(self.period[:4]), 1, 1)
            self.length = relativedelta(years=1)
        elif re.match(r"^\d{4}April$", self.period):
            self.interval = Interval.FINANCIAL_YEAR_APRIL
            self.datetime = datetime(int(self.period[:4]), 4, 1)
            self.length = relativedelta(years=1)
        elif re.match(r"^\d{4}July$", self.period):
            self.interval = Interval.FINANCIAL_YEAR_JULY
            self.datetime = datetime(int(self.period[:4]), 7, 1)
            self.length = relativedelta(years=1)
        elif re.match(r"^\d{4}Oct$", self.period):
            self.interval = Interval.FINANCIAL_YEAR_OCT
            self.datetime = datetime(int(self.period[:4]), 10, 1)
            self.length = relativedelta(years=1)
        else:
            return NotImplementedError(f"Cannot parse DHIS2 period {self.period}.")


def get_range(start: Period, end: Period) -> typing.Sequence[Period]:
    """Expand the period towards another end period.

    The function returns a range of Period objects with a frequency
    that depends on the identified interval (e.g. year, month, week).

    The frequency of the range will be determined by the interval type
    of the starting period.

    Parameters
    ----------
    period_end : Period
        The end period.

    Returns
    -------
    list of Period
        Output range of periods.

    Examples
    --------
    >>> get_range(Period("2020Q3"), Period("2021Q1"))
    [Period("2020Q3"), Period("2020Q4"), Period("2021Q1")]
    """
    if start.interval == Interval.DAY:
        return _range_day(start, end)
    elif start.interval == Interval.WEEK:
        return _range_week(start, end)
    elif start.interval == Interval.WEEK_SUNDAY:
        return _range_week_sunday(start, end)
    elif start.interval == Interval.MONTH:
        return _range_month(start, end)
    elif start.interval == Interval.BI_MONTH:
        return _range_bi_month(start, end)
    elif start.interval == Interval.QUARTER:
        return _range_quarter(start, end)
    elif start.interval == Interval.SIX_MONTH:
        return _range_six_month(start, end)
    elif start.interval == Interval.SIX_MONTH_APRIL:
        return _range_six_month_april(start, end)
    elif start.interval == Interval.YEAR:
        return _range_year(start, end)
    elif start.interval == Interval.FINANCIAL_YEAR_APRIL:
        return _range_financial_year_april(start, end)
    elif start.interval == Interval.FINANCIAL_YEAR_JULY:
        return _range_financial_year_july(start, end)
    elif start.interval == Interval.FINANCIAL_YEAR_OCT:
        return _range_financial_year_oct(start, end)
    else:
        raise ValueError("Unsupported interval type.")


def _range_day(start: Period, end: Period) -> typing.Sequence[Period]:
    drange = []
    day = start.datetime
    while day < end.datetime + end.length:
        drange.append(day.strftime("%Y%m%d"))
        day += start.length
    return drange


def _range_week(start: Period, end: Period) -> typing.Sequence[Period]:
    drange = []
    day = start.datetime
    while day < end.datetime + end.length:
        drange.append(day.strftime("%YW%W"))
        day += start.length
    return drange


def _range_week_sunday(start: Period, end: Period) -> typing.Sequence[Period]:
    drange = []
    day = start.datetime
    while day < end.datetime + end.length:
        drange.append(day.strftime("%YSunW%U"))
        day += start.length
    return drange


def _range_month(start: Period, end: Period) -> typing.Sequence[Period]:
    drange = []
    day = start.datetime
    while day < end.datetime + end.length:
        drange.append(day.strftime("%Y%m"))
        day += start.length
    return drange


def _range_bi_month(start: Period, end: Period) -> typing.Sequence[Period]:
    drange = []
    day = start.datetime
    while day < end.datetime + end.length:
        drange.append(day.strftime("%Y%mB"))
        day += start.length
    return drange


def _range_quarter(start: Period, end: Period) -> typing.Sequence[Period]:
    drange = []
    day = start.datetime
    while day < end.datetime + end.length:
        quarter = math.ceil(day.month / 3)
        drange.append(f"{day.year}Q{quarter}")
        day += start.length
    return drange


def _range_six_month(start: Period, end: Period) -> typing.Sequence[Period]:
    drange = []
    day = start.datetime
    while day < end.datetime + end.length:
        semi = math.ceil(day.month / 6)
        drange.append(f"{day.year}S{semi}")
        day += start.length
    return drange


def _range_six_month_april(start: Period, end: Period) -> typing.Sequence[Period]:
    drange = []
    day = start.datetime
    while day < end.datetime + end.length:
        semi = math.ceil((day.month - 3) / 6)
        if semi == 0:
            semi = 2
        drange.append(f"{day.year}AprilS{semi}")
        day += start.length
    return drange


def _range_year(start: Period, end: Period) -> typing.Sequence[Period]:
    return [str(y) for y in range(start.datetime.year, end.datetime.year + 1)]


def _range_financial_year_april(start: Period, end: Period) -> typing.Sequence[Period]:
    return [f"{y}April" for y in range(start.datetime.year, end.datetime.year + 1)]


def _range_financial_year_july(start: Period, end: Period) -> typing.Sequence[Period]:
    return [f"{y}July" for y in range(start.datetime.year, end.datetime.year + 1)]


def _range_financial_year_oct(start: Period, end: Period) -> typing.Sequence[Period]:
    return [f"{y}Oct" for y in range(start.datetime.year, end.datetime.year + 1)]
