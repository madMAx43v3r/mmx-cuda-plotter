#!/bin/bash

cd $(dirname "$0")

vnxcppcodegen --cleanup generated/ mmx interface/
