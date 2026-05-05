## Semi-Conductor
[[#Semi-conductor]] is a material with electrical conductivity between a conductor and an insulator.

They are usually made of silicon which has $4$ out of $8$ valence electrons. Hence, we can form silicon lattice by attaching silicon atoms to each other.

Covalent bonds are formed, electrons are shared and each silicon gets to have a full outer layer. This makes it an insulator.

To make it a conductor, we add in impurities.

---
### Impurities
We can either toss in a Boron(3 valence electrons) or a Phosphorous(5 valence electrons) into the lattice.

Tossing in Boron results in empty spaces for the electrons
- This creates a `p-type`(positive)
- Holes bump into each other, so they move

Tossing in Phosphorous results in extra electrons.
- This creates a `n-type`(negative)
- Free electrons bump into each other, so they move

---
### PN-Junctions
When P and N-types come into contact with each other, this creates a PN junction.

P-type material has holes, and n-type materials have free electrons. Free electrons closest to p-type will occupy holes.
Note that they are still being brought back as they still have protons back in the n-type to be attracted to.

The **depletion region** represents the area where the holes have been filled, and where the electrons have left.

- The region with filled holes is negative.
  There is now more electrons than protons.
- The region where electrons just left is positive.
  Protons still remain, so more protons than electrons.

This is known as **diffusion**.

---
### Drift
When enough electrons fill the empty spots, the positive region becomes positive enough to the point it starts attracting the electrons that just left back.
This is called **drift**.

Initially drift isn't strong enough to attract all electrons back.
But drift will strengthen as more electrons leave.
This happens till an equilibrium is reached, when the **diffusion** and **drift** effects are the same.

So, we now essentially have
- Pure silicon (does not conduct)
- [[#Impurities]] added (conducts again)
- [[#PN-Junctions|P and N]] put together (no conduction anymore)

---
### Forward Bias

![image|300](https://notes-media.kthiha.com/Semi-Conductor/b198211afc0e8d951025f353f0316566.png)
If we drive negative charge into the [[#PN-Junctions|n-type]] and positive charge into the [[#PN-Junctions|p-type]], 
- More electrons enter the n-type region.
  Hence, more electrons to move to holes in the p-type region.
  This **shrinks** the depletion region.
- Electrons in the [[#PN-Junctions|p-type region]] move back to the [[#PN-Junctions|n-type]] through the wire as it is attracted to positive charge.
  This creates more holes in the [[#PN-Junctions|p-type]] region.
- This induces a current in the opposite direction of electron flow.

This is called [[#forward bias]] and permits a current.

---
### Reverse Bias
If we apply positive voltage to the [[#PN-Junctions|n-type]], and negative voltage is applied to the [[#PN-Junctions|p-type]], 
- more electrons go to the [[#PN-Junctions|p-type]] region.
- this large depletion region prevents current

---
## See Also
- [[Electricity]]
- [[Semiconductor]]
- [[MOSFET]]
- [[Capacitor]]

