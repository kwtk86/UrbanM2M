from m2m_core.FoM import calc_fom
print('')
for year in range(2012, 2018):
    fom = calc_fom(f'd:/zzh2022/15.convlstm/a11_lstmca/land_{year-1}-r.tif',
             f'd:/zzh2022/15.convlstm/a11_lstmca/land_{year}-r.tif',
             f'd:/zzh2022/15.convlstm/a11_lstmca//lstmca-sim-{year}.tif')
    #
    fom2 = calc_fom(f'd:/zzh2022/15.convlstm/data-gisa-dwq/year/land_{year-1}r.tif',
             f'd:/zzh2022/15.convlstm/data-gisa-dwq/year/land_{year}r.tif',
             f'd:/zzh2022/15.convlstm/data-gisa-dwq/sim2/r8gba2-e31-s--land_{year}.tif')
    print(year, 'lstm:', round(fom,4), 'convlstm:', round(fom2, 4))