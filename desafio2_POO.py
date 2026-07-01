#ACTIVIDAD POO
#Definimos la clase Persona aplicando encapsulamiento y polimorfismo

class Persona:

    #constructor de la clase
    def __init__(self, nombre, edad):
        self.__nombre = nombre
        self.__edad = edad

    #getter para obtener el nombre
    @property
    def nombre(self):
        return self.__nombre

    #setter para modificar el nombre
    @nombre.setter
    def nombre(self, nuevo_nombre):
        if nuevo_nombre != "":
            self.__nombre = nuevo_nombre
        else:
            print("El nombre no puede estar vacío")

    #getter para obtener la edad
    @property
    def edad(self):
        return self.__edad

    #setter para modificar la edad
    @edad.setter
    def edad(self, nueva_edad):
        if nueva_edad > 0:
            self.__edad = nueva_edad
        else:
            print("La edad debe ser mayor a 0")

    #metodo saludar
    def saludar(self):
        return f"Hola, mi nombre es {self.__nombre} y tengo {self.__edad} años"

    #metodo despedir
    def despedir(self):
        return f"{self.__nombre} se despide, hasta luego"


#creamos una clase Estudiante que hereda de Persona
class Estudiante(Persona):

    def __init__(self, nombre, edad, carrera):
        super().__init__(nombre, edad)
        self.carrera = carrera

    #polimorfismo: modificamos el metodo saludar
    def saludar(self):
        return f"Hola, soy {self.nombre}, tengo {self.edad} años y estudio {self.carrera}"

    #polimorfismo: modificamos el metodo despedir
    def despedir(self):
        return f"{self.nombre} termina de estudiar y se despide"


#creamos una clase Profesor que hereda de Persona
class Profesor(Persona):

    def __init__(self, nombre, edad, ramo):
        super().__init__(nombre, edad)
        self.ramo = ramo

    #polimorfismo: modificamos el metodo saludar
    def saludar(self):
        return f"Buenos días, soy el profesor {self.nombre} y enseño {self.ramo}"

    #polimorfismo: modificamos el metodo despedir
    def despedir(self):
        return f"El profesor {self.nombre} finaliza la clase y se despide"


#creamos objetos
persona1 = Persona("Catalina", 24)
estudiante1 = Estudiante("Catalina", 24, "Ingeniería Física")
profesor1 = Profesor("Juan", 45, "Programación Orientada a Objetos")


#imprimimos los saludos
print(persona1.saludar())
print(estudiante1.saludar())
print(profesor1.saludar())

#imprimimos las despedidas
print(persona1.despedir())
print(estudiante1.despedir())
print(profesor1.despedir())


#probamos acceder a los atributos usando getters
print(persona1.nombre)
print(persona1.edad)


#modificamos los atributos usando setters
persona1.nombre = "Catalina Gómez"
persona1.edad = 25

print(persona1.saludar())


#probamos valores incorrectos
persona1.nombre = "Francis Rocha"
persona1.edad = -5