#include <iostream>
#include "ray_funcs.h"

int res_x, res_y, num_sph, num_lights, num_planes, num_mesh;
tira::camera cam;
glm::vec3 cam_pos{}, cam_look{}, cam_up{}, cam_side{};
tira::simplemesh mesh;
spheres s[100], bound_sph;
lights l[100];
plane p;

//float test_intersect(ray r, spheres s)
//{
//    int b, c, flag = 0;
//    b = glm::dot(r.dir,s.cen - r.origin);
//    c = glm::dot(s.cen - r.origin, s.cen - r.origin) - s.rad*s.rad;
//    if (b * b - c < 0)
//    {d
//        //std::cout << "b = "<<b<<"   c= "<<c;
//    }
//    else
//    {
//        flag = 1;
//        //std::cout << c;
//    }
//    return flag;
//}

void block_divide(int thread_id, int num_threads, int num_blocks, tira::image<unsigned char>& img)
{   
    int block_idx, block_row, block_col;    
    glm::vec3 color;
    for (block_idx = thread_id; block_idx < num_blocks; block_idx += num_threads)
    {
        block_row = (block_idx/10) * num_blocks;
        block_col = (block_idx%10) * num_blocks;
        if(block_row < res_x && block_col <res_y)
            raytrace_loop_cpu(block_row, block_col, img);
    }
}
    
void cuda_trace(int num_threads, tira::camera cam, tira::triangle* tri, spheres* s1, spheres bound_sph, lights* l1, plane p, int res_x, int res_y, int tri_count, int num_sph, int num_lights, int num_planes, int num_mesh, float* image_mat);

int main(int argc, char* argv[])
{

    //tira::parser image("basic.scene");
    //tira::parser image("test.scene");
    //tira::parser image("spheramid.scene");
    //std::string image_name = argv[0];
    //int num_threads = int(argv[1]);
    std::string image_name = argv[2];
    int num_threads = std::atoi(argv[3]);
    //int num_threads = 10;
    tira::parser image(image_name);
    int k, u, v, i, j;
    float back_r, back_g, back_b, bound_radius, dist, color, color_r, color_g, color_b, min_dist;
    plane tri_p;
    glm::vec3 bound_centroid, tri_int;
    res_x = image.get<int>("resolution", 0);
    res_y = image.get<int>("resolution", 1);
    back_r = image.get<float>("background", 0);
    back_g = image.get<float>("background", 1);
    back_b = image.get<float>("background", 2);
    tira::triangle *tri;
    std::string file_name;
    tira::image<unsigned char> img(res_x, res_y, 3);

    //Background
    for (u = 0; u < res_x; u++)
        for (v = 0; v < res_y; v++)
        {
            img(u, v, 0) = 255 * back_r;
            img(u, v, 1) = 255 * back_g;
            img(u, v, 2) = 255 * back_b;
        }
    //Spheres
    num_sph = image.count("sphere");
    for (k = 0; k < num_sph; k++)
    {
        s[k].rad = image.get<float>("sphere", k, 0);
        s[k].cen.x = image.get<float>("sphere", k, 1);
        s[k].cen.y = image.get<float>("sphere", k, 2);
        s[k].cen.z = image.get<float>("sphere", k, 3);
        s[k].r = image.get<float>("sphere", k, 4);
        s[k].g = image.get<float>("sphere", k, 5);
        s[k].b = image.get<float>("sphere", k, 6);
    }

    //Lights
    num_lights = image.count("light");
    for (k = 0; k < num_lights; k++)
    {
        l[k].pos.x = image.get<float>("light", k, 0);
        l[k].pos.y = image.get<float>("light", k, 1);
        l[k].pos.z = image.get<float>("light", k, 2);
        l[k].r = image.get<float>("light", k, 3);   
        l[k].g = image.get<float>("light", k, 4);
        l[k].b = image.get<float>("light", k, 5);
    }


    //Camera
    float cam_fov;
    cam_pos.x = image.get<float>("camera_position", 0);
    cam_pos.y = image.get<float>("camera_position", 1);
    cam_pos.z = image.get<float>("camera_position", 2);
    cam_look.x = image.get<float>("camera_look", 0);
    cam_look.y = image.get<float>("camera_look", 1);
    cam_look.z = image.get<float>("camera_look", 2);
    cam_up.x = image.get<float>("camera_up", 0);
    cam_up.y = image.get<float>("camera_up", 1);
    cam_up.z = image.get<float>("camera_up", 2);
    cam_side = glm::cross(cam_look, cam_up);
    cam_fov = image.get<float>("camera_fov", 0);

    cam.position(cam_pos);
    cam.lookat(cam_look);
    cam.fov(cam_fov);
    cam.up(cam_up);

    //Plane

    num_planes = image.count("plane");
    if (num_planes)                             //if planes exist on the scene file
    {
        p.point.x = image.get<float>("plane", 0);
        p.point.y = image.get<float>("plane", 1);
        p.point.z = image.get<float>("plane", 2);
        p.normal.x = image.get<float>("plane", 3);
        p.normal.y = image.get<float>("plane", 4);
        p.normal.z = image.get<float>("plane", 5);
        p.r = image.get<float>("plane", 6);
        p.g = image.get<float>("plane", 7);
        p.b = image.get<float>("plane", 8);
    }

    //Mesh
    num_mesh = image.count("mesh");
    if (num_mesh)
    {
        file_name = image.get<std::string>("mesh", 0);
        mesh.load(file_name);
        mesh.boundingsphere(bound_centroid, bound_radius);
        bound_sph.cen = bound_centroid;
        bound_sph.rad = bound_radius;
    }

    /*ray pixel_ray;
    float t, t_min, shadow_r, shadow_g, shadow_b, shadow;
    glm::vec3 c{}, plane_int{}, p_dist{}, test;
    int flag = 0,min_k;*/
    float x, y;
    float* image_mat; 
    auto start = std::chrono::system_clock::now();
    int n, m, deviceCount = 0, tri_count;
    int* testing;
    std::vector<std::thread> threads(num_threads);
    int num_blocks = 100, thread_id;
    glm::vec3 col1;
    int i1, j1;

    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess)
    {
        std::cout << "CUDA Runtime Error: " << cudaGetErrorString(err);
        return -1;
    }

    //CUDA LINKING
    if (deviceCount != 0)           //Change to ==0 for CPU version
    {
        std::cout << "CUDA Device Found! \n";
        image_mat = new float[res_x * res_y * 3];
        tri_count = mesh.count();
        tri = new tira::triangle[tri_count];

        if (!image_mat || !tri) {
            std::cerr << "Memory allocation failed!" << std::endl;
            exit(EXIT_FAILURE);
        }

        for (i = 0; i < tri_count; ++i) {
            tri[i] = mesh[i];
        }

        cuda_trace(num_threads, cam, tri, s, bound_sph, l, p, res_x, res_y, tri_count, num_sph, num_lights, num_planes, num_mesh, image_mat);

        for (i1 = 0; i1 < res_x * res_y * 3; i1++)
        {
            img(res_x - i1 / (res_y * 3) - 1, res_y - i1 % (res_y * 3) / 3 - 1, i1 % 3) = image_mat[i1];
        }


        delete[] image_mat;
        delete[] tri;
    }

    //IF NO CUDA
    else
    {
        std::cout << "CUDA Device not Found! \n";
        for (thread_id = 0; thread_id < num_threads; thread_id++)
        {
            threads[thread_id] = std::thread(block_divide, thread_id, num_threads, num_blocks, std::ref(img));
        }
        for (thread_id = 0; thread_id < num_threads; thread_id++)
            threads[thread_id].join();
    }

    img.save("output.bmp");
    auto time = std::chrono::system_clock::now() - start;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time);
    double avg_time = static_cast<double>(duration.count())/(res_x*res_y);
    std::cout << "Time spent to create the image is: " << duration.count() << " milliseconds";
    std::cout << "\n Average time per pixel is: " << avg_time << " milliseconds";

    return 0;
}