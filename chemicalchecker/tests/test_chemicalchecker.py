#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `chemicalchecker` package."""

import pytest


from chemicalchecker import chemicalchecker


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
