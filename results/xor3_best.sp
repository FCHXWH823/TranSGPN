* Best placement from 20 trials
* Objective: 139  CPP=18  Mismatch=0  WL=57  Density=6  Gaps=2
* Checkpoint: checkpoints/transnet_physical_xor_v1.pt

.SUBCKT TRANSSYN_NANBC_NABNC_ANB a b c a_N b_N c_N ZN VCC GND
* PDN — NMOS_VTL — !f(x)  [8 transistors]
MN0  nd2    a     GND    GND  NMOS_VTL  W=0.090000U  L=0.050000U
MN1  nd2    a_N   ZN     GND  NMOS_VTL  W=0.090000U  L=0.050000U
MN2  nd3    b_N   GND    GND  NMOS_VTL  W=0.090000U  L=0.050000U
MN3  nd3    b     ZN     GND  NMOS_VTL  W=0.090000U  L=0.050000U
MN4  nd3    c_N   nd2    GND  NMOS_VTL  W=0.090000U  L=0.050000U
MN5  nd4    b     GND    GND  NMOS_VTL  W=0.090000U  L=0.050000U
MN6  nd4    b_N   ZN     GND  NMOS_VTL  W=0.090000U  L=0.050000U
MN7  nd4    c     nd2    GND  NMOS_VTL  W=0.090000U  L=0.050000U
* PUN — PMOS_VTL — f(!x)  [8 transistors]
MP0  np2    a     VCC    VCC  PMOS_VTL  W=0.090000U  L=0.050000U
MP1  np2    a_N   ZN     VCC  PMOS_VTL  W=0.090000U  L=0.050000U
MP2  np3    b_N   VCC    VCC  PMOS_VTL  W=0.090000U  L=0.050000U
MP3  np3    b     ZN     VCC  PMOS_VTL  W=0.090000U  L=0.050000U
MP4  np3    c_N   np2    VCC  PMOS_VTL  W=0.090000U  L=0.050000U
MP5  np4    b     VCC    VCC  PMOS_VTL  W=0.090000U  L=0.050000U
MP6  np4    b_N   ZN     VCC  PMOS_VTL  W=0.090000U  L=0.050000U
MP7  np4    c     np2    VCC  PMOS_VTL  W=0.090000U  L=0.050000U
.ENDS
