import xml.dom.minidom as d
import trimesh
import os


def get_com_and_moi(obj_path):
    mesh = trimesh.load(obj_path)
    return mesh.center_mass, mesh.moment_inertia


class URDFHandler(object):

    def __init__(self, template_path):
        self.root = d.parse(template_path).documentElement
        self.contact = self.root.getElementsByTagName('contact')[0]
        self.inertial = self.root.getElementsByTagName('inertial')[0]
        self.visual = self.root.getElementsByTagName('visual')[0]
        self.collision = self.root.getElementsByTagName('collision')[0]

    def set_obj_name(self, obj_name):
        self.root.setAttribute('name', obj_name)

    def set_center_of_mass(self, xyz=None):
        if xyz is None:
            xyz = [0, 0, 0]
            print("[Warning] Set center of mass as [0, 0, 0], it might be wrong!")
        xyz = [str(_) for _ in xyz]
        center = self.inertial.getElementsByTagName('origin')[0]
        center.setAttribute('xyz', " ".join(xyz))

    def set_moment_of_inertia(self, moi=None):
        """ Moment of inertia is a 3x3 matrix, but it is symmetric.
        So we only need 6 number tu represent it.
        | ixx ixy ixz |
        | ixy iyy iyz |
        | ixz iyz izz |
        """
        if moi is None:
            ixx, ixy, ixz, iyy, iyz, izz = 1e-3, 0, 0, 1e-3, 0, 1e-3
        else:
            ixx, ixy, ixz, iyy, iyz, izz = moi[0][0], moi[0][1], moi[0][2], \
                                            moi[1][0], moi[0][1], moi[2][2]

        inertia = self.inertial.getElementsByTagName('inertia')[0]
        inertia.setAttribute('ixx', str(ixx))
        inertia.setAttribute('ixy', str(ixy))
        inertia.setAttribute('ixz', str(ixz))
        inertia.setAttribute('iyy', str(iyy))
        inertia.setAttribute('iyz', str(iyz))
        inertia.setAttribute('izz', str(izz))

    def set_obj(self, element, obj_name):
        if element == "v":
            geometry = self.visual.getElementsByTagName('geometry')[0]
        else:
            geometry = self.collision.getElementsByTagName('geometry')[0]
        mesh = geometry.getElementsByTagName('mesh')[0]
        mesh.setAttribute('filename', obj_name)

    def get_URDF_output(self):
        return self.root.toxml()


class URDFBuilder(object):

    def __init__(self, template_path):
        self.handler = URDFHandler(template_path)

    def obj2urdf(self, work_path, visual_name, collision_name):

        com, moi = get_com_and_moi(work_path + "/" + visual_name)
        obj_name = work_path.split("/")[-1]
        self.handler.set_obj_name(obj_name)
        self.handler.set_center_of_mass(com)
        self.handler.set_moment_of_inertia(moi)
        self.handler.set_obj("v", visual_name)
        self.handler.set_obj("c", collision_name)

        urdf_name = obj_name + ".urdf"
        urdf_path = work_path + "/" + urdf_name
        with open(urdf_path, "w") as f:
            f.write(self.handler.get_URDF_output())


if __name__ == '__main__':
    template_path = "/home/josep/code/python/rlcode/robovat/models/_prototype.urdf"
    work_space = "/home/josep/code/python/rlcode/robovat/models/urdfs"
    visual_name = "visual.obj"
    collisionName = "collision.obj"
    obj_dirs = [_ for _ in os.listdir(work_space)]

    builder = URDFBuilder(template_path)

    for i in range(len(obj_dirs)):
        builder.obj2urdf(work_space + "/" + obj_dirs[i], "visual.obj", "collision.obj")
